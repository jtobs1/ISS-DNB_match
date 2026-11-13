import os, sys, glob, argparse, subprocess
import numpy as np

"""
Wrapper for the DEGRADED (and ISS) surface terms:
    1) propgen:  particle file (.part) -> optical property file (.prp)
    2) shdom90:  .prp + surface term   -> .radout, _f1/_f2/_f4.out, .log
    3) realized tau: column optical depth map from the .prp (true, non-delta-M)
                     plus SHDOM's delta-M scaled "TAU----------" print from the log
    4) plots:    TOA radiance (.radout), upward / downward-diffuse flux
                 cross sections (_f4.out), and the realized tau map

The propgen / shdom90 inputs mirror src_single/lop_propgen_testing.csh and
run_propgen.sh, but the stdin is built here (same thing ./put does: one
argument per line) so DX/DY can be read from the surface file instead of
hardcoding 0.742.

MEMORY / MPI:
    propgen is serial and fully ALLOCATE'd -- the big .part files are fine.
    shdom90 (built with UseMPI=no) caps its memory at 1.75*HUGE(int32) bytes
    (~3.5 GB) no matter what MAX_TOTAL_MB is, because its array indices are
    32-bit. A 196x241x25 grid at Nmu=16/Nphi=32 with NUM_SH_TERM_FACTOR=1.0
    does NOT fit ("MAX_TOTAL_MB memory limit exceeded with just base grid").
    The preflight check below reproduces SHDOM's memory math so this fails in
    a second instead of after the property file is read. Options:
        --nmu 8 --nphi 16                  (NLM 256 -> 64, fits easily)
        --sh-factor 0.3                    (fewer SH terms allocated; SHDOM stops
                                            mid-run if the solution needs more)
        --mpi N --shdom-exe ./shdom90_mpi  (each process gets its own ~3.5 GB;
                                            needs an MPI build, see README notes)

Usage:
    python run_degraded_shdom.py <sfc_file> [--part <part_file>] [options]
    e.g. python run_degraded_shdom.py ../src_single/degraded_s019
         python run_degraded_shdom.py ../src_single/degraded_s019 --nmu 8 --nphi 16
         python run_degraded_shdom.py ../src_single/degraded_s019 --steps tau,plot
    Long runs: nohup python run_degraded_shdom.py ... > degraded_s019.nohup 2>&1 &

Created: Jackson Tobin 10/07/2026
"""

SRC_DIR = '/Users/jacksontobin/Local_Documents/coding_shenanigans/SINGLE_PIXEL_SIMULATIONS/src_single'
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, SIM_DIR)
# sys.stdout.reconfigure(line_buffering=True)  # keep prints in order with propgen/shdom output


# =============================================================================
# File readers
# =============================================================================
def read_sfc_header(sfc_file):
    """Returns NX, NY, DX, DY from the 2nd line of an SHDOM surface file."""
    with open(sfc_file) as f:
        f.readline()  # surface type ('L')
        dims = f.readline().split()
    return int(dims[0]), int(dims[1]), float(dims[2]), float(dims[3])


def read_prp_dims(prp_file):
    """Returns NX, NY, NZ from line 2 of the .prp (what the awk getline does)."""
    with open(prp_file) as f:
        f.readline()
        return [int(v) for v in f.readline().split()[:3]]


def parse_prp_column_tau(prp_file):
    """
    Same as run_dnb_shdom_process.parse_prp_column_tau(), but for EVERY
    column instead of just (IX=1, IY=1), since the degraded aerosol layer is
    NOT horizontally uniform. Returns the physical (not delta-M scaled) column
    optical depth map tau[IX, IY] using SHDOM's trapezoidal rule, plus the
    extinction cube ext[IX, IY, IZ] and zlevels.
    Includes the (small) Rayleigh contribution from raylcoef.
    """
    with open(prp_file) as f:
        f.readline()  # header text line

        # Tokens can wrap across lines in the header, so stream them
        tokens = []
        def take(n):
            while len(tokens) < n:
                tokens.extend(f.readline().split())
            out = tokens[:n]
            del tokens[:n]
            return out

        NX, NY, NZT = [int(v) for v in take(3)]
        take(2)  # DELX, DELY
        zlevels = np.array([float(v) for v in take(NZT)])
        nphaseout = int(take(1)[0])
        for _ in range(nphaseout):
            nl = int(take(1)[0])
            take(nl)  # skip Legendre coefficients
        assert len(tokens) == 0, 'Leftover header tokens -- unexpected .prp layout'

        # Remaining lines: IX IY IZ TEMP EXTINCT SSALB IPHASE
        data = np.loadtxt(f, usecols=(0, 1, 2, 4))

    ix, iy, iz = (data[:, :3].astype(int) - 1).T
    ext = np.zeros((NX, NY, NZT))
    ext[ix, iy, iz] = data[:, 3]

    dz = np.diff(zlevels)
    tau = np.sum(dz * (ext[:, :, :-1] + ext[:, :, 1:]) / 2.0, axis=2)
    return tau, ext, zlevels


def parse_log_tau(logfile):
    """All of EDDRTF's delta-M scaled 'TAU----------' prints (one per column)."""
    taus = []
    with open(logfile) as lf:
        for line in lf:
            if 'TAU----------' in line:
                taus.append(float(line.split()[-1]))
    return np.array(taus)


def read_f4(fname):
    """
    Reads a format-4 flux file (every base grid point). Grid indices come
    from the unique X/Y/Z values (avoids the dx-rounding issues in
    plot_flux2dv2.py / plot_rad2dv2.py).
    Returns x, y, z [km] and UP, DOWN_DIFFUSE, DOWN_DIRECT[IX, IY, IZ].
    """
    data = np.loadtxt(fname, comments='!')
    x, ix = np.unique(np.round(data[:, 0], 3), return_inverse=True)
    y, iy = np.unique(np.round(data[:, 1], 3), return_inverse=True)
    z, iz = np.unique(np.round(data[:, 2], 3), return_inverse=True)
    out = []
    for col in (3, 4, 5):
        arr = np.full((len(x), len(y), len(z)), np.nan)
        arr[ix, iy, iz] = data[:, col]
        out.append(arr)
    return x, y, z, out[0], out[1], out[2]


def read_radout(fname):
    """
    Reads a .radout file. Returns the radiance level Z and a list of
    (MU, PHI, x, y, RAD[IX, IY]) for each output direction.
    """
    blocks, zrad = [], None
    MU = PHI = None
    rows = []

    def close_block():
        if MU is None or len(rows) == 0:
            return
        arr = np.array(rows)
        x, ix = np.unique(np.round(arr[:, 0], 3), return_inverse=True)
        y, iy = np.unique(np.round(arr[:, 1], 3), return_inverse=True)
        RAD = np.full((len(x), len(y)), np.nan)
        RAD[ix, iy] = arr[:, 2]
        blocks.append((MU, PHI, x, y, RAD))

    with open(fname) as f:
        for line in f:
            cols = line.split()
            if len(cols) == 0:
                continue
            if 'RADIANCE AT Z=' in line:
                zrad = float(line.split('Z=')[1].split()[0])
            elif '(mu,phi)' in line:
                close_block()
                MU, PHI = float(cols[1]), float(cols[2])
                rows = []
            elif not cols[0].startswith('!'):
                rows.append([float(c) for c in cols[:3]])
    close_block()
    return zrad, blocks


# =============================================================================
# SHDOM memory preflight (mirrors shdom90.f90 lines ~148-320)
# =============================================================================
def shdom_memory_check(NX, NY, NZ, Nmu, Nphi, BCflag, IPflag, maxmb,
                       adapt, sh_factor, cell_ratio, splitacc, accel=True):
    ML = Nmu - 1
    MM = max(0, Nphi // 2 - 1)
    # Night runs: SOLARAZ=0 but IPflag bit 1 is off, so NCS=2
    NPHI0MAX = Nphi
    NLM = (2*MM + 1)*(ML + 1) - MM*(MM + 1)
    NX1 = NX if (IPflag & 1 or BCflag & 1 or BCflag & 4) else NX + 1
    NY1 = NY if (IPflag & 2 or BCflag & 2 or BCflag & 8) else NY + 1
    NBPTS = NX1*NY1*NZ
    MEMWORD = Nmu*(2 + 2*Nphi + 2*NLM + 2*33*32) + 4.5*NX*NY*NZ

    huge = 2**31 - 1
    if maxmb*1024.0**2 > 1.75*huge:
        maxmb = 1.75*huge/1024.0**2
    if splitacc <= 0.0:
        adapt = 1.0
    big_arrays = 3 if accel else 2
    per_pt = 28 + 16.5*cell_ratio + NPHI0MAX + sh_factor*NLM*big_arrays
    wantmem = adapt*NBPTS*per_pt
    reduce = min(1.0, ((maxmb*1024**2)/4 - MEMWORD)/wantmem)
    adapt_eff = adapt*reduce

    print('==================================')
    print(f'SHDOM MEMORY PREFLIGHT: NBPTS={NBPTS:,}  NLM={NLM}  NPHI0MAX={NPHI0MAX}')
    print(f'  words/point={per_pt:.1f}  MAX_TOTAL_MB used={maxmb:.0f}')
    print(f'  ADAPT_GRID_FACTOR {adapt} -> {adapt_eff:.2f}  (must be >= 1)')
    print('==================================')
    return adapt_eff >= 1.0, adapt_eff


# =============================================================================
# Steps 1 and 2: propgen + shdom90
# =============================================================================
def make_mie_table(scatfile, wavelen1, wavelen2):
    """Smoke Mie table, same settings as lop_propgen_testing.csh."""
    stdin = '\n'.join([
        f'{wavelen1} {wavelen2}', 'A', '(1.635,-0.004)', '1.8', 'L', '0.4',
        '40 0.1 4.0', '10.0', scatfile]) + '\n'
    subprocess.run('./make_mie_table', input=stdin, text=True, cwd=SRC_DIR, check=True)


def run_propgen(partfile, prpfile, scatfile, raylcoef):
    # Same ordering as: ./put $#scattables $scattables "$scattypes" $parfile
    #   $maxnewphase $asymtol $fracphasetol $raylcoef $Nzother $ZTother $prpfile
    scattables = [scatfile]
    scattypes = '1'
    maxnewphase, asymtol, fracphasetol = 100, 0.01, 0.1
    Nzother, ZTother = 0, []  # Height(km)/Temp(K) pairs of extra z levels
    stdin = '\n'.join([str(len(scattables)), *scattables, scattypes, partfile,
                       str(maxnewphase), str(asymtol), str(fracphasetol), str(raylcoef),
                       str(Nzother), *[str(v) for v in ZTother], prpfile]) + '\n'
    print('==================================')
    print(f'PROPGEN: {partfile} -> {prpfile}')
    print('==================================')
    subprocess.run('./propgen', input=stdin, text=True, cwd=SRC_DIR, check=True)


def run_shdom(args, prpfile, sfcfile, outbase, DX, DY, wavelen):
    Nx, Ny, Nz = read_prp_dims(os.path.join(SRC_DIR, prpfile))

    numout = 4
    mu0, phi0, flux0 = 1.0, 0.0, 0.0
    BCflag, IPflag, DeltaM = args.bcflag, 0, 'T'
    shacc = 0.0
    solacc, accel, maxiter = 1.0E-4, 'T', 100
    # Routparm = (Z dx dy ? ? n_angles mu_1 phi_1 ...); spacing = actual grid spacing
    Routparm = [str(args.rad_z), f'{DX}', f'{DY}', '0.0', '0.0', str(len(args.mu))]
    for m in args.mu:
        Routparm += [str(m), '0.0']
    Foutparm = f'2 2.000 {DX} {DY}'

    ok, _ = shdom_memory_check(Nx, Ny, Nz, args.nmu, args.nphi, BCflag, IPflag,
                               args.maxmb, args.adapt, args.sh_factor, args.cell_ratio,
                               args.splitacc)
    if not ok and args.mpi <= 1:
        sys.exit('Base grid does not fit in serial shdom90 -- try --nmu 8 --nphi 16, '
                 'a smaller --sh-factor, or --mpi N (see docstring).')

    # Same ordering as the ./put ... | ./shdom90 call in run_propgen.sh
    stdin = [os.path.basename(outbase), prpfile, sfcfile, 'NONE', 'NONE', 'NONE',
             f'{Nx} {Ny} {Nz}', f'{args.nmu} {args.nphi}',
             str(BCflag), str(IPflag), DeltaM, 'P', 'S', f'{flux0} {mu0} {phi0}', '0.0',
             str(wavelen), f'{args.splitacc} {shacc}', f'{accel} {solacc} {maxiter}',
             str(numout),
             'R', *Routparm, f'{outbase}.radout',
             'F', '1', f'{outbase}_f1.out',
             'F', Foutparm, f'{outbase}_f2.out',
             'F', '4', f'{outbase}_f4.out',
             'NONE', str(args.maxmb), str(args.adapt), str(args.sh_factor), str(args.cell_ratio)]
    stdin = '\n'.join(stdin) + '\n'

    cmd = [args.shdom_exe]
    if args.mpi > 1:
        cmd = ['mpirun', '-np', str(args.mpi), args.shdom_exe]

    logfile = os.path.join(SRC_DIR, f'{outbase}.log')
    print('==================================')
    print(f'SHDOM: {" ".join(cmd)}')
    print(f'  PROPERTY FILE: {prpfile}')
    print(f'  SURFACE FILE:  {sfcfile}')
    print(f'  Nx={Nx} Ny={Ny} Nz={Nz} | Nmu={args.nmu} Nphi={args.nphi} | DX={DX} DY={DY}')
    print(f'  LOG: {logfile}')
    print('==================================')

    # Tee stdout to the log, but don't echo the one-per-column TAU prints
    with open(logfile, 'w') as lf:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, cwd=SRC_DIR)
        proc.stdin.write(stdin)
        proc.stdin.close()
        for line in proc.stdout:
            lf.write(line)
            if 'TAU----------' not in line:
                print(line, end='', flush=True)
        proc.wait()
    if proc.returncode != 0:
        sys.exit(f'shdom90 exited with code {proc.returncode} -- see {logfile}')


# =============================================================================
# Step 3: realized tau
# =============================================================================
def realized_tau(outbase):
    prpfile = os.path.join(SRC_DIR, f'{outbase}.prp')
    logfile = os.path.join(SRC_DIR, f'{outbase}.log')

    tau_prp, _, _ = parse_prp_column_tau(prpfile)
    np.save(os.path.join(SRC_DIR, f'realized_tau_map_{outbase}.npy'), tau_prp)

    tau_log = parse_log_tau(logfile) if os.path.exists(logfile) else np.array([])
    if tau_log.size == 0:
        print(f'WARNING: no TAU prints in {logfile} -- only the .prp tau is available')

    def stats(a):
        return (np.nanmin(a), np.nanmean(a), np.nanmax(a)) if a.size else (None,)*3

    csv = os.path.join(SRC_DIR, f'realized_tau_{outbase}.csv')
    with open(csv, 'w') as f:
        f.write('source,min,mean,max\n')
        f.write('tau_prp,{},{},{}\n'.format(*stats(tau_prp)))
        f.write('tau_shdom,{},{},{}\n'.format(*stats(tau_log)))
    print(f'tau_prp   (true column OD)  min/mean/max = {stats(tau_prp)}')
    print(f'tau_shdom (delta-M scaled)  min/mean/max = {stats(tau_log)}')
    print(f'Saved {csv}')
    return tau_prp


# =============================================================================
# Step 4: plots
# =============================================================================
def plot_all(args, outbase, DX, DY, tau_prp=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from image_wrapper import wrap_image as wrap

    plotdir = os.path.join(SIM_DIR, 'degraded_plots', outbase)
    os.makedirs(plotdir, exist_ok=True)

    def norm(arr, vmax=None):
        good = arr[np.isfinite(arr) & (arr > 0)]
        top = vmax if vmax is not None else (np.nanmax(arr) if good.size else 1.0)
        if args.log and good.size:
            return colors.LogNorm(vmin=max(good.min(), top*1e-6), vmax=top)
        return colors.Normalize(vmin=0., vmax=top)

    # NOTE: images are drawn with X horizontal, Y vertical and origin='lower'
    # (km extents), unlike plot_rad2dv2.py's imshow(np.flipud(RAD)).
    def extent(a, b, da, db):
        return [a[0] - da/2, a[-1] + da/2, b[0] - db/2, b[-1] + db/2]

    # ---------------- Realized tau map ----------------
    if tau_prp is not None:
        NX, NY = tau_prp.shape
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(tau_prp.T, origin='lower', cmap='viridis',
                       extent=[0, NX*DX, 0, NY*DY])
        fig.colorbar(im, ax=ax, shrink=0.7, label='column optical depth (.prp)')
        ax.set_xlabel('X [km]'); ax.set_ylabel('Y [km]')
        ax.set_title(f'Realized tau | {outbase}')
        plt.tight_layout()
        plt.savefig(os.path.join(plotdir, f'{outbase}_tau_prp.png'), dpi=200)
        plt.close()

    # ---------------- TOA radiances ----------------
    radfile = os.path.join(SRC_DIR, f'{outbase}.radout')
    if os.path.exists(radfile):
        zrad, blocks = read_radout(radfile)
        for MU, PHI, x, y, RAD in blocks:
            if MU != 1.0:
                RAD = wrap(RAD, MU, PHI, dx=DX, dy=DY, h=args.rad_z)
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(RAD.T, origin='lower', cmap='cubehelix',
                           norm=norm(RAD, args.rad_vmax), extent=extent(x, y, DX, DY))
            fig.colorbar(im, ax=ax, shrink=0.7, label='[W/(m^2 um sr)]')
            ax.set_xlabel('X [km]'); ax.set_ylabel('Y [km]')
            ax.set_title(f'TOA (z={zrad}km) Radiance | {outbase} | MU={MU}, PHI={PHI}')
            plt.tight_layout()
            out = os.path.join(plotdir, f'{outbase}_rad_{np.round(MU, 2)}.png')
            plt.savefig(out, dpi=200)
            plt.close()
            print(f'Saved {out}')
    else:
        print(f'WARNING: no {radfile}')

    # ---------------- Flux cross sections (f4) ----------------
    f4file = os.path.join(SRC_DIR, f'{outbase}_f4.out')
    if not os.path.exists(f4file):
        print(f'WARNING: no {f4file}')
        return
    x, y, z, UP, DOWN_DIFFUSE, DOWN_DIRECT = read_f4(f4file)
    dz = np.diff(z).mean() if len(z) > 1 else 1.0

    # Default slices go through the brightest surface upward flux
    imax, jmax = np.unravel_index(np.nanargmax(UP[:, :, 0]), UP[:, :, 0].shape)
    xz_slices = args.xz if args.xz else [jmax]
    yz_slices = args.yz if args.yz else [imax]
    xy_levels = args.xy if args.xy else [0, len(z)//2, len(z) - 1]

    def flux_fig(up, dd, ext, xlabel, ylabel, title, out, aspect):
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(9, 9))
        for ax, arr, name, cmap, vmax in ((axs[0], up, 'Upward', 'magma', args.up_vmax),
                                          (axs[1], dd, 'Downward Diffuse', 'viridis', args.down_vmax)):
            im = ax.imshow(arr, origin='lower', cmap=cmap, norm=norm(arr, vmax),
                           extent=ext, aspect=aspect)
            # fig.colorbar(im, ax=ax, shrink=0.8, label='[W/(m^2 um)]')
            fig.colorbar(im, ax=ax, shrink=0.8, label='ISS Nikon D3 Measure')
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.set_title(f'{name} Flux | {title}')
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()
        print(f'Saved {out}')

    for j in xz_slices:
        flux_fig(UP[:, j, :].T, DOWN_DIFFUSE[:, j, :].T, extent(x, z, DX, dz),
                 'X [km]', 'Z [km]', f'xz at Y={y[j]:.2f} km (j={j})',
                 os.path.join(plotdir, f'{outbase}_flux_xz_{j:03d}.png'), 'auto')
    for i in yz_slices:
        flux_fig(UP[i, :, :].T, DOWN_DIFFUSE[i, :, :].T, extent(y, z, DY, dz),
                 'Y [km]', 'Z [km]', f'yz at X={x[i]:.2f} km (i={i})',
                 os.path.join(plotdir, f'{outbase}_flux_yz_{i:03d}.png'), 'auto')
    for k in xy_levels:
        flux_fig(UP[:, :, k], DOWN_DIFFUSE[:, :, k], extent(x, y, DX, DY),
                 'X [km]', 'Y [km]', f'xy at Z={z[k]:.2f} km (k={k})',
                 os.path.join(plotdir, f'{outbase}_flux_xy_{k:03d}.png'), 'equal')


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='propgen + shdom90 + realized tau + plots for degraded surface terms')
    p.add_argument('sfc', help='surface term file (e.g. ../src_single/degraded_s019)')
    p.add_argument('--part', help='particle file (default: <sfc>.part)')
    p.add_argument('--outbase', help='output base name in src_single (default: <sfc name>_w070)')
    p.add_argument('--steps', default='propgen,shdom,tau,plot',
                   help='comma list of: propgen, shdom, tau, plot')
    p.add_argument('--force', action='store_true', help='rerun propgen/shdom even if outputs exist')
    # SHDOM parameters
    p.add_argument('--mu', type=float, nargs='+', default=[1.0], help='radiance output mu(s)')
    p.add_argument('--rad-z', type=float, default=25.0, help='radiance output height [km]')
    p.add_argument('--nmu', type=int, default=16)
    p.add_argument('--nphi', type=int, default=32)
    p.add_argument('--splitacc', type=float, default=0.01)
    p.add_argument('--bcflag', type=int, default=0)
    p.add_argument('--maxmb', type=float, default=3500.)
    p.add_argument('--adapt', type=float, default=2.5, help='ADAPT_GRID_FACTOR')
    p.add_argument('--sh-factor', type=float, default=1.0, help='NUM_SH_TERM_FACTOR')
    p.add_argument('--cell-ratio', type=float, default=1.5, help='CELL_TO_POINT_RATIO')
    p.add_argument('--mpi', type=int, default=1, help='number of MPI processes (needs an MPI build)')
    p.add_argument('--shdom-exe', default='./shdom90', help='relative to src_single')
    # Plot options
    p.add_argument('--xz', type=int, nargs='*', help='Y indices for xz flux slices (default: brightest)')
    p.add_argument('--yz', type=int, nargs='*', help='X indices for yz flux slices (default: brightest)')
    p.add_argument('--xy', type=int, nargs='*', help='Z indices for xy flux slices (default: bottom/mid/top)')
    p.add_argument('--log', action='store_true', help='log color scales')
    p.add_argument('--rad-vmax', type=float)
    p.add_argument('--up-vmax', type=float)
    p.add_argument('--down-vmax', type=float)
    args = p.parse_args()

    steps = args.steps.split(',')
    sfc = os.path.abspath(args.sfc)
    part = os.path.abspath(args.part) if args.part else sfc + '.part'

    wavename = '070'
    wavelen1 = wavelen2 = wavelen = 0.70
    raylcoef = 0.00124  # 0.70um: (2.97E-4)*lambda^(-4.15+0.2*lambda)
    scatfile = f'smoke_w{wavename}.scat'

    outbase = args.outbase or f'{os.path.basename(sfc)}_w{wavename}'
    prpfile = f'{outbase}.prp'

    # Relative paths from src_single: propgen/shdom90 have short filename buffers
    sfc_rel = os.path.relpath(sfc, SRC_DIR)
    part_rel = os.path.relpath(part, SRC_DIR)

    NX, NY, DX, DY = read_sfc_header(sfc)
    print(f'SURFACE: {sfc_rel}  NX={NX} NY={NY} DX={DX} DY={DY}')
    if DX > 5.0 or DY > 5.0:
        print(f'WARNING: DX={DX}, DY={DY} km per pixel (domain {NX*DX:.0f} x {NY*DY:.0f} km).')
        print('         If these are meters, fix the surface/particle headers first --')
        print('         SHDOM treats DELX/DELY as km.')

    if 'propgen' in steps:
        if not os.path.exists(part):
            sys.exit(f'No particle file {part} -- make it with part_from_sfc.py first.')
        if not os.path.exists(os.path.join(SRC_DIR, scatfile)):
            make_mie_table(scatfile, wavelen1, wavelen2)
        if args.force or not os.path.exists(os.path.join(SRC_DIR, prpfile)):
            run_propgen(part_rel, prpfile, scatfile, raylcoef)
        else:
            print(f'SKIPPING propgen: {prpfile} exists (use --force)')

    if 'shdom' in steps:
        if args.force or not os.path.exists(os.path.join(SRC_DIR, f'{outbase}.radout')):
            run_shdom(args, prpfile, sfc_rel, outbase, DX, DY, wavelen)
        else:
            print(f'SKIPPING shdom90: {outbase}.radout exists (use --force)')

    tau_prp = None
    if 'tau' in steps:
        tau_prp = realized_tau(outbase)

    if 'plot' in steps:
        taufile = os.path.join(SRC_DIR, f'realized_tau_map_{outbase}.npy')
        if tau_prp is None and os.path.exists(taufile):
            tau_prp = np.load(taufile)
        plot_all(args, outbase, DX, DY, tau_prp)
