import sys
import numpy as np
from numba import njit, objmode
from scipy.ndimage import uniform_filter # type: ignore
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed # type: ignore # For parallel processing
from scipy.ndimage import rotate as rotate
"""
Purpose:
    Iteratively "degrade" (downscale) ISS imagery to DNB resolution
    and convert a surface term for each degradation.
    This script builds off "scale_analysis.ipynb".

Created: Jackson Tobin 08/20/2026
"""

def degraded_image(s, im1):
    """Follows degraded_process(), but only returns the degraded image."""
    im = (im1 - np.nanmean(im1)).astype(np.float64) # remove NANs
    blurred = uniform_filter(im, size=s, mode='reflect') 
    deg = blurred[::s, ::s]
    return deg

# Super speedy version!
def degraded_process(s, w, im1, max_offsets=8, plots=False, DX=None, DY=None):
    ts = datetime.datetime.now()
    im = (im1 - np.nanmean(im1)).astype(np.float64) # remove NANs
    blurred = uniform_filter(im, size=s, mode='reflect') # blur once

    # Get the unique offsets in both X and Y
    offsets = np.unique(np.linspace(0, s, min(s, max_offsets), endpoint=False).astype(int))
    vals = []
    for dy in offsets:
        for dx in offsets:                  # includes diagonals
            deg = blurred[dy::s, dx::s]     # degraded image at step (dx, dy)
            
            # Make the surface term ==================================
            make_sfcTrm(deg, f'./sfcTrms/degraded_dx{dx:03d}_dy{dy:03d}_s{s:03d}', DX=DX,DY=DY)

            if min(deg.shape) < 4 * w:      # Skip if its too small
                continue
            m = uniform_filter(deg, size=w, mode='reflect')
            sq = uniform_filter(deg**2, size=w, mode='reflect')
            vals.append(np.mean(np.maximum(sq - m**2, 0))) # Average Loc. Var. 

    vals = np.array(vals)
    # Display elapsed time
    elapsed = datetime.datetime.now() - ts
    print(f"Time for {s} w/ val {vals.mean()}: {elapsed}")
    
    if plots:
        if vals.size == 0:
            return np.nan, np.nan, blurred[::s, ::s]
        return vals.mean(), vals.std(), blurred[::s, ::s]
    else:
        if vals.size == 0:
            return np.nan, np.nan, blurred[::s, ::s].size
        return vals.mean(), vals.std(), blurred[::s, ::s].size

def make_sfcTrm(im, filename, DX, DY):
    """
    Creates the SHDOM Surface Term (.txt file).
    Must alter DX, DY to account for spatial extent.
    """
    # Save in SHDOM surface term format:
    NX0, NY0 = im.shape
    # DX = (1.60934 / 212) * 1000 # image resolution from JPG and Google Earth
    # DY = (1.22310 / 199) * 1000
    with open(f'{filename}', 'w') as f:
        f.write('L\n')
        f.write(f'{NX0} {NY0} {DX} {DY}\n')
        for i in range(NX0):
            for j in range(NY0):
                IX = i + 1
                IY = j + 1
                SfcTemp = 300
                SfcAlbedo = 0.0
                SfcEmissivity = np.flipud(im)[i, j]
                SfcEmissivity = 0.0 if SfcEmissivity < 0 else SfcEmissivity
                f.write(f"{IX} {IY} {SfcTemp} {SfcAlbedo} {SfcEmissivity}\n")

def linear_transformation(src, a):
    """
    Performs a "shearing" linear transformation on an image.
    Must provide "a," which follows y=a*x...
    """
    M, N = src.shape
    points = np.mgrid[0:N, 0:M].reshape((2, M*N))
    new_points = np.linalg.inv(a).dot(points).round().astype(int)
    x, y = new_points.reshape((2, M, N), order='F')
    indices = x + N*y
    return np.take(src, indices, mode='wrap')
            
def vis0(imr):
    """
    Plots ISS imagery nicely
    """
    # Plot the images: grayscale and "spectral"
    fig, (ax1) = plt.subplots(figsize=(10, 10))
    matplotlib.rcParams['figure.dpi'] = 300
    plot1 = ax1.imshow(imr, cmap='cubehelix')
    cbar=plt.colorbar(plot1, shrink=0.5)
    plt.grid(True, linestyle='-', alpha=0.5, color='orange')
    plt.tight_layout()
    plt.show()

def calibrate_iss(iss):
    """
    Calibrates the ISS image using DNB radiances.
    Methodology still not available...
    """
    iss = (iss / np.amax(iss)) * 10
    return iss


if __name__=="__main__":
    import matplotlib.pyplot as plt
    # from scipy.ndimage import rotate
    plt.style.use('dark_background')
    from PIL import Image
    import cv2
    # matplotlib.rcParams['figure.dpi'] = 200 # Use full image resolution for display

    # Read the image =======================================
    fname = '/Users/jacksontobin/Local_Documents/coding_shenanigans/ISSim_PROCESS/ISS066-E-23917.JPG' # Fort Collins: Night
    # fname = '/Users/jacksontobin/Local_Documents/coding_shenanigans/ISSim_PROCESS/ISS072-E-498409.JPG' # Philly: Night
    im = cv2.imread(fname)
    # Flatten to a 1-color channel image
    im1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Cast as a numpy array
    im1 = np.array(im1)[500:2000,2000:3250] # Fort Collins: Night subset
    # im1 = np.array(im1)[:,1000:] # Foco: remove foothills
    im1 = (im1 / np.amax(im1)) * 10

    # Pad the image with ~75 0s
    im1 = np.pad(im1, pad_width=200)

    # Rotate the image by the JPSS inclination angle (approximate)
    phi = 90-98.7 # SNPP inclination angle
    im1 = rotate(im1, angle=phi)

    # vis0(im1)

    # Define the dimmensions ===============================
    # dx = 8.35 # +/- 0.1 [m/px] # Philly
    # dy = 8.05 # +/- 0.1 [m/px]
    dx = 7.59 / 1000 # +/- 0.1 [m] # Fort Colllins
    dy = 6.15 / 1000 # +/- 0.1 [m]

    degraded_process(97, w=3, im1=im1, max_offsets=100, plots=False, DX=dx, DY=dy)
    sys.exit()

    # Create the degraded images and Surface Terms
    # sd = np.arange(1, 337, 1) # Foco subset image
    sd = np.arange(10, 110, 1) # Foco subset image
    sd = [97]
    for s in sd:
        print(f"Creating s = {s}")
        degraded = degraded_image(s, im1)
        DX = dx * s     # Scale the pixel sizes 
        DY = dy * s
        make_sfcTrm(degraded, f'./sfcTrms/degraded_s{s:03d}', DX, DY)


