import sys
import numpy as np
import degraded_to_sfcTerm as degrade
from scipy.ndimage import rotate as rotate
from scipy.ndimage import uniform_filter # type: ignore
import matplotlib.pyplot as plt
from PIL import Image
import cv2
'''
GOAL
Match the DNB image with a shifted ISS image.
Can do this using the pre-made Surface Terms, (compute efficient)
or doing the shifting here. (compute inefficient)

METHOD
    1) Normalize the ISS and DNB radiance fields.
    2) Shift + tilt the ISS image.
    3) Find the "best match" ISS image based on (some metric).

Created: Jackson Tobin 11/13/2026
'''
# Apply the classic IDL styling rules
idl_style = {
    # Inward ticks on all 4 borders
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    
    # Tick lengths and visibility
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    
    # Thick axes lines and borders
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,
    
    # Fonts and Text (Clean Sans-Serif mimicking Helvetica/Hershey)
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 12,
    
    # Simple, high-contrast defaults (No backgrounds, grids, or transparencies)
    'axes.grid': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
}
plt.rcParams.update(idl_style)

def same_size(iss, dnb):
    """
    Ensure the ISS and DNB images are the same size,
    cover roughly the same geographical regions
    More of a debugging step:
        Will need to plot out the images to check
    """
    return None

def degrade_iss(iss, dnb, w=3, sx=None, sy=None, max_offsets=8):
    """
    Creates numpy arrays of shifted ISS "radiance" fields.
    Like "degraded_process" in "run_degraded_shdom," but does not
    create surface terms for each ISS shift.

    The ISS pixels are not square, so the block size is set separately
    for each axis.

    The ISS array passed in should extend at least one block (sy rows,
    sx cols) beyond the nominal FoCo window, so every shift (ox, oy) still
    has enough real pixels to fill dnb.shape. Each degraded image is then
    trimmed to exactly dnb.shape (no padding needed).

    Args:
        sx, sy: ISS pixels per DNB pixel in x (columns) and y (rows).
                If None, computed from the array shapes (only valid if the
                ISS array has NOT been extended by a margin).
    """
    if sx is None:
        sx = int(np.ceil(iss.shape[1] / dnb.shape[1]))
    if sy is None:
        sy = int(np.ceil(iss.shape[0] / dnb.shape[0]))
    ny, nx = dnb.shape

    iss = (iss - np.nanmean(iss)).astype(np.float64) # remove NANs
    blurred = uniform_filter(iss, size=(sy, sx), mode='reflect') # blur once, (rows, cols)
    # Get the unique offsets in X and Y (different block sizes per axis)
    offs_x = np.unique(np.linspace(0, sx, min(sx, max_offsets), endpoint=False).astype(int))
    offs_y = np.unique(np.linspace(0, sy, min(sy, max_offsets), endpoint=False).astype(int))
    for oy in offs_y:
        for ox in offs_x: # includes diagonals
            # degraded image at offset (ox, oy), trimmed to the DNB shape
            deg = blurred[oy::sy, ox::sx][:ny, :nx]
            if deg.shape != dnb.shape:
                raise ValueError(f'Degraded ISS {deg.shape} smaller than DNB {dnb.shape} '
                                 f'at shift ({ox},{oy}); crop a larger ISS margin.')
            #Renormalize
            deg = deg / deg.max()

            # if oy == 1:
            #     sys.exit()
            # im=plt.imshow(deg, cmap='grey', vmin=0, vmax=1.0)
            # plt.colorbar(im, shrink=0.5)
            # plt.title(f'Normalized ISS at DNB Spatial Resolution; Shift ({ox},{oy}) \nNo Radiometric Calibration')
            # plt.savefig(f'./test_iss_dx{ox:03d}_dy{oy:03d}.png', dpi=100)
            # plt.close()

def rotate_iss(iss):
    # Rotate the ISS image by the inclination angle
    phi = 90-98.7 # SNPP inclination angle
    return rotate(iss, angle=phi)

# ---------------------------------------------------------------------------
# Georeferencing: anchor the ISS photo to the ground, then average it over
# each DNB pixel's footprint. Replaces the (ox, oy) shift search.
# ---------------------------------------------------------------------------

def lonlat_to_km(lon, lat, lon0, lat0):
    """
    Convert lon/lat [deg] to local (planar) east/north distances [km] from (lon0, lat0).
    """
    x = (lon - lon0) * 111.32 * np.cos(np.radians(lat0))  # east  [km]
    y = (lat - lat0) * 110.57                              # north [km]
    return x, y

def fit_iss_homography(tie_points, lon0, lat0):
    """
    Fit the mapping  ground (east, north km) -> ISS pixel (col, row).
    Args:
        tie_points: list of (iss_col, iss_row, lon, lat) in the FULL ISS image.
                    At least 4; ideally 6-10 spread across the city.
    Returns:
        H: 3x3 homography matrix
    """
    tp = np.asarray(tie_points, dtype=np.float64)
    pix = tp[:, :2]                                   # x, y pixels (col, row)
    ground = np.column_stack(lonlat_to_km(tp[:, 2], tp[:, 3], lon0, lat0)) # shape ~= each tp
    H, _ = cv2.findHomography(ground, pix, method=0)  # least squares on all points

    # Quality check: where does each tie point land vs. where you picked it?
    pred = cv2.perspectiveTransform(ground.reshape(-1, 1, 2), H).reshape(-1, 2)
    resid = np.hypot(*(pred - pix).T) # "*" unpacks array.
    for k, r in enumerate(resid):
        print(f'tie point {k}: residual {r:6.1f} ISS px')
    return H

def dnb_pixel_corners(a):
    """
    Corner values of each DNB pixel from its center values (lat or lon).
    (ny, nx) centers -> (ny+1, nx+1) corners. Each interior corner is the
    average of the 4 surrounding centers; the outer edge is extrapolated by
    half a pixel. Corner [i, j] is the top-left corner of pixel [i, j].
    """
    a = np.vstack([2*a[0] - a[1], a, 2*a[-1] - a[-2]])                    # extend rows
    a = np.hstack([2*a[:, :1] - a[:, 1:2], a, 2*a[:, -1:] - a[:, -2:-1]])  # extend cols
    return 0.25 * (a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:])

def iss_on_dnb_grid(iss_full, H, lat_sub, lon_sub, lon0, lat0, shift=(0.0, 0.0)):
    """
    Average the ISS pixels inside each DNB pixel footprint.
    Returns an array the same shape as the DNB subset (NaN where a
    footprint falls outside the ISS photo).

    Args:
        shift: (fx, fy) move every DNB footprint on the ground by fx pixels
               along the DNB column (scan) direction and fy pixels along the
               DNB row (track) direction. (0, 0) = VIIRS geolocation as-is.
    """
    ny, nx = lat_sub.shape
    # Footprint corners: lon/lat -> km
    x, y = lonlat_to_km(dnb_pixel_corners(lon_sub), dnb_pixel_corners(lat_sub), lon0, lat0)
    # Apply the sub-pixel shift on the ground, in units of DNB pixels.
    # col_step / row_step = average ground vector [km] from one DNB pixel to the next.
    col_step = np.array([np.diff(x, axis=1).mean(), np.diff(y, axis=1).mean()])
    row_step = np.array([np.diff(x, axis=0).mean(), np.diff(y, axis=0).mean()])
    fx, fy = shift
    x = x + fx * col_step[0] + fy * row_step[0]
    y = y + fx * col_step[1] + fy * row_step[1]

    # km -> ISS pixel coords (col, row)
    ground = np.stack([x, y], axis=-1).reshape(-1, 1, 2).astype(np.float64)
    pix = cv2.perspectiveTransform(ground, H).reshape(ny+1, nx+1, 2)

    out = np.full((ny, nx), np.nan)
    SHIFT = 4  # fillPoly sub-pixel precision: coordinates in 1/2**4 ISS px
    for i in range(ny):
        for j in range(nx):
            # 4 corners of DNB pixel [i, j], in order around the quadrilateral
            quad = np.array([pix[i, j], pix[i, j+1], pix[i+1, j+1], pix[i+1, j]])
            c_lo, r_lo = np.floor(quad.min(axis=0)).astype(int)
            c_hi, r_hi = np.ceil(quad.max(axis=0)).astype(int) + 1
            if c_lo < 0 or r_lo < 0 or c_hi > iss_full.shape[1] or r_hi > iss_full.shape[0]:
                continue  # footprint not fully inside the photo -> leave NaN
            # Rasterize the quadrilateral into a mask over its bounding box
            mask = np.zeros((r_hi - r_lo, c_hi - c_lo), np.uint8)
            local = np.round((quad - [c_lo, r_lo]) * 2**SHIFT).astype(np.int32)
            cv2.fillPoly(mask, [local], 1, shift=SHIFT)
            out[i, j] = iss_full[r_lo:r_hi, c_lo:c_hi][mask == 1].mean()
    return out

def radiometric_fit(iss_grid, dnb):
    """
    With the geometry fixed, compare radiances: fit dnb ~ a*iss + b.
    Returns gain a, offset b, correlation r, and RMSE of the fit.
    The linear fit compares geospatial "radiance" pairs – 
    """

    ok = np.isfinite(iss_grid) & np.isfinite(dnb) # checks for nans, out-of-bounds
    a, b = np.polyfit(iss_grid[ok], dnb[ok], 1) # linear fit through scatter plot of "radiances"
    r = np.corrcoef(iss_grid[ok], dnb[ok])[0, 1] # linear correlation coefficient of "radiances"
    rmse = np.sqrt(np.mean((a*iss_grid[ok] + b - dnb[ok])**2))
    return a, b, r, rmse

def shift_sweep(iss_full, H, lat_sub, lon_sub, lon0, lat0, dnb, n=11, max_frac=0.5):
    """
    Shift the DNB footprints over the georeferenced ISS by fractions of a
    DNB pixel (-max_frac .. +max_frac in each direction, n steps) and score
    each shift with radiometric_fit.
    Returns:
        fracs: the shift values [DNB pixels]
        R, RMSE: (n, n) arrays indexed [fy, fx]
    """
    fracs   = np.linspace(-max_frac, max_frac, n)
    R       = np.zeros((n, n))
    rmse    = np.zeros((n, n))
    for iy, fy in enumerate(fracs):
        for ix, fx in enumerate(fracs):
            g = iss_on_dnb_grid(iss_full, H, lat_sub, lon_sub, lon0, lat0, shift=(fx, fy))
            _, _, R[iy, ix], rmse[iy, ix] = radiometric_fit(g, dnb)
            
            # Debugging / plotting section
            if iy == 1:
                sys.exit()
            # g /= g.max()
            # im=plt.imshow(g, cmap='grey', vmin=0, vmax=1.0)
            # plt.colorbar(mappable=im, shrink=0.8)
            # plt.xlabel('x pixel')
            # plt.ylabel('y pixel')
            # plt.title(f'Normalized ISS at DNB Spatial Resolution; Shift ({ix},{iy})')
            # plt.savefig(f'./test_iss_dx{ix:03d}_dy{iy:03d}.png', dpi=100)
            # plt.close()

    return fracs, R, rmse

def DNB_ISS_match(dnb, iss, sx=None, sy=None):
    """
    Args:
        dnb: Day/Night Band radiance field (.npy)
        iss: ISS "radiance" field (.npy)
    Returns:
        shifted_iss: the ISS image that best matches the DNB
    """
    # Normalize the DNB and ISS radiances
    dnb = dnb / dnb.max()
    iss = iss / iss.max()
    # Nominal ISS pixel sizes [km] -- NOTE: these don't match the DNB
    # geometry for this crop (fitting the shapes implies ~8.4 m rows, ~7.7 m cols),
    # so the block sizes are derived from the array shapes instead.
    dx = 7.59 / 1000 # +/- 0.1 [m] # Fort Colllins
    dy = 6.15 / 1000 # +/- 0.1 [m]
    degrade_iss(iss=iss, dnb=dnb, w=3, sx=sx, sy=sy, max_offsets=50)

if __name__ == "__main__":
    import read_VNP as vnp

    # Read in the dnb (July ...) and iss images
    fname_iss = '/Users/jacksontobin/Local_Documents/coding_shenanigans/SINGLE_PIXEL_SIMULATIONS/DNB-ISS_match/ISS066-E-23917.JPG'
    iss = cv2.imread(fname_iss) # read the image
    iss = cv2.cvtColor(iss, cv2.COLOR_BGR2GRAY) # convert to grayscale
    iss_full = np.array(iss)
    # iss = np.pad(iss, pad_width=200) # Pad the image with 0s to allow for rotation

    # TESTING DNB image – will get correct dnb file later
    f2 = "/Users/jacksontobin/Local_Documents/coding_shenanigans/SINGLE_PIXEL_SIMULATIONS/DNB-ISS_match/VNP03DNB.A2026186.0812.002.2026186162826.nc"
    f1 = "/Users/jacksontobin/Local_Documents/coding_shenanigans/SINGLE_PIXEL_SIMULATIONS/DNB-ISS_match/VNP02DNB.A2026186.0812.002.2026186164827.nc"
    vnp03 = vnp.read_VNP03DNB(f2)
    vnp02 = vnp.read_VNP02DNB(f1)
    # Mask out the -999
    cond = (vnp02.DNB_OBSERVATIONS == -999.9)
    vnp02.DNB_OBSERVATIONS[cond] = 0
    # Subset by geolocation
    latmax, lonmin = 40.62229375157484, -105.17162350202236 # top left
    latmin, lonmax = 40.474305022549544, -104.96230269066936 # bottom right
    # Apply the subset
    subset = vnp03.subset_idx(latmin,latmax,lonmin,lonmax)
    rows, cols = np.where(subset)
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()
    # Subset the latitude, longitude, radiance
    lat_sub = vnp03.LAT[row_min:row_max+1, col_min:col_max+1]
    lon_sub = vnp03.LON[row_min:row_max+1, col_min:col_max+1]
    dnb = vnp02.DNB_OBSERVATIONS[row_min:row_max+1, col_min:col_max+1]

    dnb = dnb / dnb.max()

    # ---------------- Georeferenced ISS -> DNB grid ----------------
    # Tie points: (iss_col, iss_row, lon, lat) in the FULL ISS image.
    #   iss_col/iss_row: hover over the feature in a matplotlib window
    #                    (x = col, y = row in the bottom bar)
    #   lon/lat:         right-click the same feature in Google Maps
    # PLACEHOLDER: these 4 just pin the old by-eye crop corners to the lat/lon
    # box, so the pipeline runs. Replace with 6-10 real landmarks.
    # TIE_POINTS = [
    #     (1400,  300, lonmin, latmax),
    #     (3900,  300, lonmax, latmax),
    #     (1400, 2400, lonmin, latmin),
    #     (3900, 2400, lonmax, latmin),
    # ]
    TIE_POINTS = [
        (2391, 1039, -105.08839037732287, 40.569027267586), # Canvas stadium
        (2480, 1644, -105.07994840722611, 40.52401065566397), # Fort Collins Jeep
        (2521, 857, -105.07702542408586, 40.58150362220208), # Mulberry x College
        (2096, 870, -105.1150079493755, 40.582008285115464), # BBQ Pit
        (2091, 1267, -105.11507083029744, 40.552854467818264), # Taft Safeway
        (3013, 864, -105.0327050969125, 40.58097820373376) 
    ]

    lon0, lat0 = lon_sub.mean(), lat_sub.mean()  # local km origin
    H = fit_iss_homography(TIE_POINTS, lon0, lat0)
    iss_grid = iss_on_dnb_grid(iss_full.astype(np.float64), H, lat_sub, lon_sub, lon0, lat0)
    a, b, r, rmse = radiometric_fit(iss_grid, dnb)
    print(f'dnb ~ {a:.4g}*iss + {b:.4g}   r = {r:.3f}   RMSE = {rmse:.4f}')
    # iss_grid = (iss_grid/iss_grid.max())**2.2 # Gamma scaling for JPG. Will want camera raw NEF file.
    iss_grid /= iss_grid.max()
    
    # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # im=plt.imshow((iss_full/iss_full.max())[0:2000, 1000:4000], vmin=0, vmax=1.0, cmap='grey')
    # plt.colorbar(mappable=im, label='Normalized, Greyscale Nikon Measures', shrink=0.5, pad=0.01)
    # plt.title('ISS Full Resolution', size=20)
    # plt.xlabel('X Pixel')
    # plt.ylabel('Y Pixel')
    # plt.tight_layout()
    # # plt.savefig('./iss_image.png', dpi=150)
    # plt.savefig('./iss_image.png', dpi=450)
    # plt.close()
    # sys.exit()
    
    # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # im=plt.imshow((iss_full/iss_full.max())[0:2000, 1000:4000], vmin=0, vmax=1.0, cmap='grey')
    # plt.colorbar(mappable=im, label='Normalized, Greyscale Nikon Measures', shrink=0.5, pad=0.01)
    # plt.title('ISS Full Resolution', size=20)
    # plt.xlabel('X Pixel')
    # plt.ylabel('Y Pixel')
    # plt.tight_layout()
    # # plt.savefig('./iss_image.png', dpi=150)
    # plt.savefig('./iss_image.png', dpi=450)
    # plt.close()

    # fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    # im1 = ax.imshow(a*iss_grid+b-dnb, cmap='coolwarm'); ax.set_title('DNB-ISS ')
    # plt.colorbar(mappable=im1, ax=ax)
    # # im1 = ax[1].imshow(dnb, cmap='grey'); ax[1].set_title('DNB')
    # # plt.colorbar(mappable=im1, ax=ax[1])
    # plt.savefig('./residual.png', dpi=100)
    # plt.close()

    # fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    # im1 = ax[0].imshow(iss_grid, cmap='grey'); ax[0].set_title('Coarsened ISS')
    # plt.colorbar(mappable=im1, ax=ax[0], shrink=0.5)
    # im1 = ax[1].imshow(dnb, cmap='grey');      ax[1].set_title('SNPP VIIRS DNB')
    # plt.colorbar(mappable=im1, ax=ax[1], shrink=0.5)
    # plt.tight_layout()
    # plt.savefig('./iss_on_dnb_grid.png', dpi=100)
    # plt.close()
    # sys.exit()

    # ---------------- Sub-pixel shifts of the DNB grid over the anchored ISS ----------------
    n = 97 # number of fractional shifts
    iss_f = iss_full.astype(np.float64)
    fracs, R, RMSE = shift_sweep(iss_f, H, lat_sub, lon_sub, lon0, lat0, dnb, n=n, max_frac=0.5)
    iy, ix = np.unravel_index(np.nanargmin(RMSE), RMSE.shape)
    print(f'best shift: fx = {fracs[ix]:+.2f}, fy = {fracs[iy]:+.2f} DNB px '
          f'(~{fracs[ix]*0.75:+.2f}, {fracs[iy]*0.75:+.2f})km   '
          f'r={R[iy, ix]:.3f}   RMSE={RMSE[iy, ix]:.4f}')
    # print(f'no shift:   r = {R[len(fracs)//2, len(fracs)//2]:.3f}   '
    #       f'RMSE = {RMSE[len(fracs)//2, len(fracs)//2]:.4f}')

    ext = [fracs[0], fracs[-1], fracs[-1], fracs[0]]  # [fx_min, fx_max, fy_max, fy_min]
    im = plt.imshow(RMSE, extent=ext, cmap='jet')
    plt.colorbar(im, label='RMSE')
    plt.plot(fracs[ix], fracs[iy], 'r+', ms=12, mew=2)
    plt.xlabel('fractional shift along DNB columns [DNB px]')
    plt.ylabel('fractional shift along DNB rows [DNB px]')
    plt.title('DNB grid shift over georeferenced ISS')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'./shift_sweep_rmse_n{n}.png', dpi=100)
    plt.close()

    # ---------------- Old shift-search approach (kept for reference) ----------------
    # r0, r1, c0, c1 = 300, 2400, 1400, 3900
    # sy = int(np.ceil((r1 - r0) / dnb.shape[0]))  # 88 for 2100/24
    # sx = int(np.ceil((c1 - c0) / dnb.shape[1]))  # 97 for 2500/26
    # iss = iss_full[r0:r1 + sy, c0:c1 + sx]
    # DNB_ISS_match(dnb=dnb, iss=iss, sx=sx, sy=sy)
