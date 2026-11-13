import netCDF4 
import h5py as h5
import netCDF4 as nc
import numpy as np

"""
Module for reading NASA VIIRS VNP02DNB data
NOTE: does NOT contain geolocation data! That is in VNP03DNB
classes:
    read_VNP02DNB
    read_VNP03DNB
"""



class read_VNP03DNB():
    '''
    Class to read NASA VIIRS VNP02DNB data.
    Contains:
        h5 format:
            - DNB_observations (number_of_lines x number_of_pixels)
            - DNB_quality_flags (number_of_lines x number_of_pixels)
            - ev_mid_time (number_of_scans), seconds
            - scan_end_time (number_of_scans)
            - scan_quality_flags (number_of_scans)
            - scan_start_time (number_of_scans)
            - scan_state_flags (number_of_scans)
        netCDF format:
            - 
    '''
    def __init__(self, filename):
        self.filename = filename
        self.netcdf = nc.Dataset(self.filename, 'r') # Read in as netCDF4
        self.h5file = h5.File(self.filename, 'r') # Read in as h5py
        
        # ======================================================================
        # Read H5 Data
        # ======================================================================
        # Overarching attributes
        self.GEOLOCATION_DATA = self.h5file['geolocation_data']
        self.NAVIGATION_DATA = self.h5file['navigation_data']
        self.NUMBER_OF_LINES = self.h5file['number_of_lines']
        self.NUMBER_OF_PIXELS = self.h5file['number_of_pixels']
        self.NUMBER_OF_SCANS = self.h5file['number_of_scans']
        self.QUATERNION_ELEMENTS = self.h5file['quaternion_elements']
        self.SCAN_LINE_ATTRIBUTES = self.h5file['scan_line_attributes']
        self.VECTOR_ELEMENTS = self.h5file['vector_elements']

        # Subgroups
        # Geolocation
        self.HEIGHT = self.GEOLOCATION_DATA['height'][:]
        self.LAND_WATER_MASK = self.GEOLOCATION_DATA['land_water_mask'][:]
        self.LAT = self.GEOLOCATION_DATA['latitude'][:]
        self.LON = self.GEOLOCATION_DATA['longitude'][:]
        self.LUNAR_AZIMUTH = self.GEOLOCATION_DATA['lunar_azimuth'][:]
        self.LUNAR_ZENITH = self.GEOLOCATION_DATA['lunar_zenith']
        self.MOON_ILLUMINATION_FRACTION = self.GEOLOCATION_DATA['moon_illumination_fraction']
        self.MOON_PHASE_ANGLE = self.GEOLOCATION_DATA['moon_phase_angle']
        self.QUALITY_FLAG = self.GEOLOCATION_DATA['quality_flag']
        self.RANGE = self.GEOLOCATION_DATA['range']
        self.SENSOR_AZIMUTH = self.GEOLOCATION_DATA['sensor_azimuth'][:]
        self.SENSOR_ZENITH = self.GEOLOCATION_DATA['sensor_zenith'][:]
        self.SOLAR_AZIMUTH = self.GEOLOCATION_DATA['solar_azimuth'][:]
        self.SOLAR_ZENITH = self.GEOLOCATION_DATA['solar_zenith'][:]
        # Navigation
        self.ATT_ANG_END = self.NAVIGATION_DATA['att_ang_end'][:]
        self.ATT_ANG_START = self.NAVIGATION_DATA['att_ang_start'][:]
        self.ATT_ANG_MID = self.NAVIGATION_DATA['att_ang_mid'][:]
        self.ATT_QUAT_EV = self.NAVIGATION_DATA['att_quat_ev'][:]
        self.EARTH_MOON_DISTANCE = self.NAVIGATION_DATA['earth_moon_distance'][:]
        self.EARTH_SUN_DISTANCE = self.NAVIGATION_DATA['earth_sun_distance'][:]
        self.LUNAR_INST = self.NAVIGATION_DATA['lunar_inst'][:]
        self.LUNAR_J2000 = self.NAVIGATION_DATA['lunar_j2000'][:]
        self.ORB_POS_EV_MID = self.NAVIGATION_DATA['orb_pos_ev_mid'][:]
        self.ORB_POS_EV_START = self.NAVIGATION_DATA['orb_pos_ev_start'][:]
        self.ORB_POS_EV_END = self.NAVIGATION_DATA['orb_pos_ev_end'][:]

        # Scan line attributes
        self.HAM_SIDE = self.SCAN_LINE_ATTRIBUTES["HAM_side"][:]
        self.EV_MID_TIME = self.SCAN_LINE_ATTRIBUTES['ev_mid_time'][:]
        self.SCAN_END_TIME = self.SCAN_LINE_ATTRIBUTES['scan_end_time'][:]
        self.SCAN_START_TIME = self.SCAN_LINE_ATTRIBUTES['scan_start_time'][:]
        self.SCAN_QUALITY = self.SCAN_LINE_ATTRIBUTES['scan_quality'][:]
        self.SENSOR_MODE = self.SCAN_LINE_ATTRIBUTES['sensor_mode'][:]

    def list_data(self):
        print(self.netcdf)
        def print_name(name):
            print(name)
        self.h5file.visit(print_name)

    def subset_idx(self, latmin, latmax, lonmin, lonmax):
        """
        apply a box subset to the latitudes, longitudes.
        Return the indices of the valid points
        """
        return (
            (self.LAT >= latmin) & (self.LAT <= latmax) &
            (self.LON >= lonmin) & (self.LON <= lonmax)
        )



class read_VNP02DNB():
    '''
    Class to read NASA VIIRS VNP02DNB data.
    Contains:
        h5 format:
            - DNB_observations (number_of_lines x number_of_pixels)
            - DNB_quality_flags (number_of_lines x number_of_pixels)
            - ev_mid_time (number_of_scans), seconds
            - scan_end_time (number_of_scans)
            - scan_quality_flags (number_of_scans)
            - scan_start_time (number_of_scans)
            - scan_state_flags (number_of_scans)
        netCDF format:
            - 
    '''
    def __init__(self, filename):
        self.filename = filename
        self.h5file = h5.File(self.filename, 'r')
        self.netcdf = nc.Dataset(self.filename, 'r') # netCDF4 data.

        # ======================================================================
        # Read H5 Data
        # ======================================================================
        # Overarching attributes
        self.OBSERVATION_DATA = self.h5file['observation_data']
        self.NUMBER_OF_LINES = self.h5file['number_of_lines']
        self.NUMBER_OF_PIXELS = self.h5file['number_of_pixels']
        self.NUMBER_OF_SCANS = self.h5file['number_of_scans']
        self.SCAN_LINE_ATTRIBUTES = self.h5file['scan_line_attributes']
        
        # Subgroups
        self.DNB_OBSERVATIONS = self.OBSERVATION_DATA['DNB_observations'][:]
        self.DNB_QUALITY_FLAGS = self.OBSERVATION_DATA['DNB_quality_flags'][:]
        self.EV_MID_TIME = self.SCAN_LINE_ATTRIBUTES['ev_mid_time'][:]
        self.SCAN_END_TIME = self.SCAN_LINE_ATTRIBUTES['scan_end_time'][:]
        self.SCAN_QUALITY_FLAGS = self.SCAN_LINE_ATTRIBUTES['scan_quality_flags'][:]
        self.SCAN_START_TIME = self.SCAN_LINE_ATTRIBUTES['scan_start_time'][:]
        self.SCAN_STATE_FLAGS = self.SCAN_LINE_ATTRIBUTES['scan_state_flags'][:]

    def list_data(self):
        print(self.netcdf)
        def print_name(name):
            print(name)
        self.h5file.visit(print_name)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy
    import numpy as np
    import os, glob


    f2 = "/home/jtobin/NIGHTSKY/3DRTM/TOBIN_ENV/CASE_STUDIES/VNP0203DNB/VNP03DNB.A2024124.0906.002.2024124155548.nc"
    f1 = "/home/jtobin/NIGHTSKY/3DRTM/TOBIN_ENV/CASE_STUDIES/VNP0203DNB/VNP02DNB.A2024124.0906.002.2024183193429.nc"
    vnp03 = read_VNP03DNB(f2)
    vnp02 = read_VNP02DNB(f1)

    # extent = [-98, -96, 34.5, 36]
    latmax, lonmin, latmin, lonmax = 36.68511508807103, -97.82603295427553, 36.6501798484569, -97.77498351484677
    extent = [lonmin, lonmax, latmin, latmax]

    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.pcolormesh(vnp03.LON, vnp03.LAT, vnp02.DNB_OBSERVATIONS, vmin=0, vmax=1e-7, cmap='cubehelix', transform=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.BORDERS, linestyle='--', color='white')
    ax.coastlines(color='white')
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True)
    plt.savefig('./testing_vnp.png', dpi=300)
    plt.close()

    # vnp02dnb = read_VNP02DNB('/home/jtobin/NIGHTSKY/3DRTM/TOBIN_ENV/CASE_STUDIES/VNP02DNB/VNP02DNB.A2024122.0806.002.2024183193136.nc')
    # vnp02dnb.list_data()