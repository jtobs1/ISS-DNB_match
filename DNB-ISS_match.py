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

def same_size(iss, dnb):
    """
    Ensure the ISS and DNB images are the same size,
    cover roughly the same geographical regions
    More of a debugging step:
        Will need to plot out the images to check
    """
    return None

def degrade_iss(s, w, iss, dnb, max_offsets=8, DX=None, DY=None):
    """
    Creates numpy arrays of shifted ISS "radiance" fields.
    Like "degraded_process" in "run_degraded_shdom," but does not 
    create surface terms for each ISS shift.
    """
    iss = (iss - np.nanmean(iss)).astype(np.float64) # remove NANs
    blurred = uniform_filter(iss, size=s, mode='reflect') # blur once
    # Get the unique offsets in both X and Y
    offsets = np.unique(np.linspace(0, s, min(s, max_offsets), endpoint=False).astype(int))
    for dy in offsets:
        for dx in offsets:                  # includes diagonals
            deg = blurred[dy::s, dx::s]     # degraded image at step (dx, dy)
            
            plt.imshhow(dnb)
            plt.savefig(f'./test_dnb.png')
            plt.close()
            sys.exit()
            plt.imshhow(deg)
            plt.savefig(f'./test_iss_dx{dx}_dy{dy}.png')
            plt.close()
            sys.exit()


def DNB_ISS_match(dnb, iss):
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

    # Rotate the ISS image by the inclination angle
    phi = 90-98.7 # SNPP inclination angle
    iss = rotate(iss, angle=phi)

    # Define the dimensions of the ISS image
    dx = 7.59 / 1000 # +/- 0.1 [m] # Fort Colllins
    dy = 6.15 / 1000 # +/- 0.1 [m]

    degrade_iss(97, w=3, iss=iss, dnb=dnb, max_offsets=50, DX=dx, DY=dy)

if __name__ == "__main__":
    import read_VNP as vnp

    # Read in the dnb (July ...) and iss images
    fname_iss = '/Users/jacksontobin/Local_Documents/coding_shenanigans/SINGLE_PIXEL_SIMULATIONS/DNB-ISS_match/ISS066-E-23917.JPG'
    iss = cv2.imread(fname_iss) # read the image
    iss = cv2.cvtColor(iss, cv2.COLOR_BGR2GRAY) # convert to grayscale
    iss = np.array(iss)[500:2000, 2000:3250] # FoCo nighttime subset
    iss = np.pad(iss, pad_width=200) # Pad the image with 0s to allow for rotation

    # TESTING DNB image – will get correct dnb file later
    f2 = "/home/jtobin/NIGHTSKY/3DRTM/TOBIN_ENV/CASE_STUDIES/VNP0203DNB/VNP03DNB.A2024124.0906.002.2024124155548.nc"
    f1 = "/home/jtobin/NIGHTSKY/3DRTM/TOBIN_ENV/CASE_STUDIES/VNP0203DNB/VNP02DNB.A2024124.0906.002.2024183193429.nc"
    vnp03 = vnp.read_VNP03DNB(f2)
    vnp02 = vnp.read_VNP02DNB(f1)

    plt.imshow(vnp02.DNB_OBSERVATIONS)
    plt.savefig('./dnb_testing.png')
    plt.close()

    plt.imshow(iss)
    plt.savefig('./iss_testing.png')
    plt.close()





    DNB_ISS_match()
