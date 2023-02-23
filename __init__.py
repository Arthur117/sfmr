from functools import wraps

import glob
import pyproj
from affine import Affine
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
from rasterio.features import rasterize
import rasterio as rio
import os
import re
import datetime
import logging
import pathurl
from cyclobs_utils.coloc_config import SARConfig, LBandConfig, get_lat, get_lon, logger
import numpy as np
import time
from numba.typed import Dict
from numba.core import types
from numba import jit
import numba

# Initialize memory monitor
mem_monitor = True
try:
    from psutil import Process
except ImportError:
    logger.warning("psutil module not found. Disabling memory monitor")
    mem_monitor = False

SAR_missions = ["RS2", "S1A", "S1B"]
l_band_missions = ["SMOS", "SMAP"]

# SFMR source noaa_aoml_hrd
HRD = "HRD"
# SFMR source noaa_nesdis_star
NESDIS = "NESDIS"

# Map sfmr variables depending on source to standardize access
sfmr_sources_variables = {
    NESDIS: {"time": "time", "lon": "longitude", "lat": "latitude", "wind_speed": "wind_speed", "quality": "quality"},
    HRD: {"time": "time", "lon": "LON", "lat": "LAT", "wind_speed": "SWS", "quality": "FLAG"}
}

# Map sfmr variables to new name in final product
sfmr_var_mapping = {
    NESDIS: {
        "wind_speed": "sfmr_wind_speed",
        "rain_rate": "sfmr_rain_rate",
        "altitude": "sfmr_altitude",
        "time": "sfmr_time",
        # "translation_speed": "sfmr_translation_speed",
        "longitude": "sfmr_lon",
        "latitude": "sfmr_lat"
    },
    HRD: {
        "SWS": "sfmr_wind_speed",
        "LON": "sfmr_lon",
        "LAT": "sfmr_lat",
        # "translation_speed": "sfmr_translation_speed",
        "RALT": "sfmr_altitude",
        "RANG": "sfmr_roll_angle",
        "PANG": "sfmr_pitch_angle",
        "ATEMP": "sfmr_air_temperature",
        "SST": "sfmr_sea_surface_temperature",
        "SALN": "sfmr_salinity",
        "SRR": "sfmr_rain_rate",
        "FWS": "sfmr_flight_level_wind_speed",
        "FDIR": "sfmr_flight_level_wind_direction",
        "time": "sfmr_time"
    }
}
# Helper dict to find source sfmr variable name from final name
sfmr_var_mapping_reversed = {
    HRD: dict((v, k) for k, v in sfmr_var_mapping[HRD].items()),
    NESDIS: dict((v, k) for k, v in sfmr_var_mapping[NESDIS].items())
}

# SFMR variables excluded from resampling
sfmr_time_resample_excluded = ["sfmr_time", "sfmr_lon", "sfmr_lat"]


def timing(f):
    """provide a @timing decorator for functions, that log time spent in it"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        mem_str = ''
        process = None

        if mem_monitor:
            process = Process(os.getpid())
            startrss = process.memory_info().rss

        starttime = time.time()
        result = f(*args, **kwargs)
        endtime = time.time()

        if mem_monitor:
            endrss = process.memory_info().rss
            mem_str = 'mem: %+.1fMb' % ((endrss - startrss) / (1024 ** 2))

        logger.info('timing %s : %.1fs. %s' % (f.__name__, endtime - starttime, mem_str))
        return result

    return wrapper


def sfmr_data_source(sfmr_ds):
    """
    Find SFMR source type from xarray dataset and return string indicating the source. Global variables NESDIS and
    HRD are used.

    Parameters
    ----------
    sfmr_ds : xarray.Dataset
              SFMR xarray dataset

    Returns
    -------
    str
        String indicating the SFMR source
    """
    if "institution" in sfmr_ds.attrs:
        if NESDIS in sfmr_ds.attrs["institution"]:
            return NESDIS
    elif "Source" in sfmr_ds.attrs:
        if HRD in sfmr_ds.attrs["Source"]:
            return HRD


def get_sat_acq_time(sat_ds, mission_config):
    """
    Retrieve acquisition time from dataset attributes and converts it to python datetime.

        Parameters
        ----------
    sat_ds : xarray.Dataset
            xarray dataset of the satellite data
    mission_config: MissionConfig
                    instance of MissionConfig for the current satellite acquisition

    Returns
    -------
    datetime.datetime
        Satellite acquisition timestamp.

    """

    acq_time = sat_ds.attrs[mission_config.acquisition_time_attribute]

    dt_acq_time = datetime.datetime.strptime(acq_time, mission_config.datetime_format)

    return dt_acq_time


@timing
def sfmr_resample(sfmr_df, sfmr_resample_freq_sec, sfmr_lon, excluded_vars):
    """
    Apply a rolling mean on the SFMR track. The row count is preserved. The mean is applied on all dataframe
    variables except those specified in parameter excluded_vars.

    Parameters
    ----------
    sfmr_df : pd.DataFrame
              Dataframe containing SFMR data with time index (which is not averaged)
    sfmr_resample_freq_sec : seconds to resample to.
    sfmr_lon : str
               Column name in SFMR dataframe to access longitude data.
    excluded_vars : list of str
                    Variables in SFMR dataframe that should not be averaged.

    Returns
    -------
    pd.DataFrame
        DataFrame of SFMR data with averaged data.
    """
    # sfmr_df = sfmr_df.resample(f"{mission_config.sfmr_resample_freq_sec}S").mean()
    # sfmr_df = sfmr_df.reset_index()
    # window_len = 10
    # for index, row in sfmr_df.iterrows():
    #    s_i = int(index-window_len/2)
    #    if s_i < 0:
    #        s_i = 0
    #    e_i = int(index+window_len/2)
    #    # sfmr_wnd_smooth[i] = np.nanmean(sfmr_wnd[s_i:e_i+1])
    #    # sfmr_rr_smooth[i] = np.nanmean(sfmr_rr[s_i:e_i+1])
    #    row["wind_speed"] = sfmr_df["wind_speed"].iloc[s_i:e_i+1].mean()
    #    #sfmr_rr_smooth[i] = sfmr_rr[s_i:e_i+1].mean()
    #
    ## The resample can create NaNs. Removing them
    # sfmr_df = sfmr_df[~pd.isna(sfmr_df[sfmr_lon])]
    # sfmr_df = sfmr_df.set_index("time")
    for col in sfmr_df.columns:
        if col not in excluded_vars:
            sfmr_df[col] = sfmr_df[col].rolling(
                sfmr_resample_freq_sec, center=True).mean()

    return sfmr_df


@jit(nopython=True)
def spatial_downsample_numba(sfmr_vars, sfmr_time, sfmr_lat_col, sfmr_lon_col, downsample_dist):
    """
    Downsample spatialy the SFMR data. Numba is used to speed up the process. The number of rows is likely to be
    reduced.

    Parameters
    ----------
    sfmr_vars : numba.typed.Dict(str: float64[:])
                Numba dict containing SFMR data (except time) as 1D numpy arrays of type float64
    sfmr_time : numpy.array of float64
                Array containing SFMR timestamp in format "seconds since UNIX epoch"
    sfmr_lat_col : str
                   Name of sfmr column to access latitude data
    sfmr_lon_col : str
                   Name of sfmr column to access longitude data
    downsample_dist : float
                      Spatial distance in degrees used to group pixels that are averaged together.

    Returns
    -------
    np.array(dtype=float64), numba.typed.Dict(str: float64[:])
        Array containing SFMR timestamp in format "seconds since UNIX epoch" and Numba dict containing SFMR data
        (except time) as 1D numpy arrays of type float64
    """
    meaned_time = np.empty(0, dtype=types.float64)

    meaned_vars = dict()
    for var in sfmr_vars:
        meaned_vars[var] = np.empty(0, dtype=types.float64)

    n = 0
    sfmr_lon = sfmr_vars[sfmr_lon_col]
    sfmr_lat = sfmr_vars[sfmr_lat_col]
    while n < sfmr_time.shape[0]:
        dis = 0.
        s = 0
        while dis < downsample_dist and (n + s) < sfmr_time.shape[0]:
            dis = np.sqrt((sfmr_lon[n] - sfmr_lon[n + s]) ** 2. +
                          (sfmr_lat[n] - sfmr_lat[n + s]) ** 2.)
            s = s + 1

        meaned_time = np.append(meaned_time, np.array(sfmr_time[n:(n + s)].mean()))
        for var in sfmr_vars:
            meaned_vars[var] = np.append(meaned_vars[var], sfmr_vars[var][n:(n + s)].mean())

        n = n + s

    return meaned_time, meaned_vars


@timing
def spatial_downsample(sfmr_df, sfmr_lat, sfmr_lon, sfmr_time, spatial_downsample_dist, sfmr_source):
    """
    Prepare spatial downsample using numba.

    Parameters
    ----------
    sfmr_df : pd.DataFrame
              DataFrame containing SFMR data with datetime as index.
    sfmr_lat : str
                Name of sfmr column to access latitude data
    sfmr_lon : str
                Name of sfmr column to access longitude data
    sfmr_time : str
                Name of sfmr time index
    mission_config : MissionConfig
                     MissionConfig instance for the current processed satellite.
    sfmr_source : str
                  String indicating the SFMR data source.

    Returns
    -------
    pd.DataFrame
        DataFrame containing SFMR data with datetime as index.
    """

    # Dict that'll contain sfmr_vars. We use Numba dict type to make it work inside @jit function.
    sfmr_vars = Dict.empty(
        key_type=types.string,
        value_type=types.float64[:]
    )

    # Populating the numba dict using SFMR variables
    for var in sfmr_var_mapping[sfmr_source]:
        if var != sfmr_time:
            sfmr_vars[var] = sfmr_df[var].to_numpy(dtype="float64")

    # Converting time to "since UNIX epoch" format because numba doesn't do well with datetimes.
    sfmr_timestamp = ((sfmr_df.index - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')).astype("float64").to_numpy()

    # Computing spatial downsampling
    meaned_time, meaned_vars = spatial_downsample_numba(sfmr_vars, sfmr_timestamp, sfmr_lat, sfmr_lon,
                                                        spatial_downsample_dist)

    # Rebuilding dataframe
    sfmr_df = pd.DataFrame(data=dict(meaned_vars), index=meaned_time)

    # Reformatting the time as datetime type
    sfmr_df.index = pd.to_datetime(sfmr_df.index, unit="s")

    # The mean operation for the downsampling generated milliseconds : removing them.
    sfmr_df.index = sfmr_df.index.round("S")
    sfmr_df.index.name = sfmr_time

    return sfmr_df

    # sfmr_df = sfmr_df.reset_index()
    # new_sfmr_df = pd.DataFrame(columns=sfmr_df.columns)
    # n = 0
    # while n < len(sfmr_df.index):
    #    dis = 0.
    #    s = 0
    #    while dis < mission_config.sfmr_spatial_downsample_dist and (n + s) < len(sfmr_df.index):
    #        dis = np.sqrt((sfmr_df.iloc[n][sfmr_lon] -
    #                       sfmr_df.iloc[n + s][sfmr_lon]) ** 2. + \
    #                      (sfmr_df.iloc[n][sfmr_lat] -
    #                       sfmr_df.iloc[n + s][sfmr_lat]) ** 2.)
    #        s = s + 1
    #
    #    t_m = sfmr_df.iloc[n:(n + s)][sfmr_time].mean()
    #    m = sfmr_df.iloc[n:(n + s)].mean()
    #    m[sfmr_time] = t_m
    #    new_sfmr_df = new_sfmr_df.append(m, ignore_index=True)
    #    # new_sfmr_df = pd.concat([new_sfmr_df, sfmr_df.iloc[n:(n+s)].mean()])
    #    n = n + s
    #
    # new_sfmr_df[sfmr_time] = new_sfmr_df[sfmr_time].apply(
    #    lambda x: datetime.datetime(x.year, x.month, x.day, x.hour, x.minute, x.second))
    # new_sfmr_df = new_sfmr_df.set_index([sfmr_time])
    #
    # return new_sfmr_df


@timing
def sfmr_translate(sfmr_df, track_df, acq_time, sfmr_lon, sfmr_lat, sfmr_time):
    """
    Modify longitude/latitude of each SFMR point to adapt it to the satellite acquisition by removing
    the cyclone translation during the SFMR flight.

    Parameters
    ----------
    sfmr_df : pd.DataFrame
              DataFrame containing SFMR data with datetime as index.
    track_df : pd.DataFrame
               DataFrame containing cyclone track data.
    acq_time : datetime.datetime
               Satellite acquisition datetime.
    sfmr_lon : str
                Name of sfmr column to access longitude data
    sfmr_lat : str
                Name of sfmr column to access latitude data
    sfmr_time : str
                Name of sfmr time index

    Returns
    -------
    pd.DataFrame
        DataFrame containing SFMR data with datetime as index with translated lon/lat.
    """
    lon = "lon"
    lat = "lat"

    logger.debug("Translating SFMR data...")
    # track_df_ts = track_df.drop(columns=["geometry", "date"])
    track_df_ts = track_df.set_index("datetime")

    # Upsample track data to know the position of cyclone every second.
    track_df_ts = track_df_ts.resample('1S').asfreq().interpolate()

    # Get cyclone position at time of satellite acquisition
    cyc_pos_sat = track_df_ts.loc[acq_time][[lon, lat]]

    logger.debug(f"Cyclone position at satellite acquisition time : {cyc_pos_sat}")

    sfmr_df = sfmr_df.reset_index()
    # Compute cyclone position at each sfmr acquisition time
    cyc_pos_sfmr = track_df_ts.loc[sfmr_df[sfmr_time]][[lon, lat]]

    logger.debug(f"Cyclone position at each sfmr acquisition time : {cyc_pos_sfmr}")

    # Compute the lon/lat distance between the cyclone position at time of acquisition and
    # cyclone position for each SFMR point
    pos_diff = pd.DataFrame()
    pos_diff[lon] = cyc_pos_sat[lon] - cyc_pos_sfmr[lon]
    pos_diff[lat] = cyc_pos_sat[lat] - cyc_pos_sfmr[lat]
    pos_diff = pos_diff.reset_index()
    logger.debug(f"Position diff : {pos_diff}")

    # Adding the offsets to for each SFMR point
    # The goal is to relocalize each SFMR point correctly into the instant-acquisition data lon/lat grid.
    # If the cyclone has moved 5 meters on the right between the time of SFMR point acquisition and the time of SAR
    # acquisition, we move the SFMR point 5 meter to the right. Thus, we can consider that this new offseted point
    # has been acquired at the same time as the SAR acquisition.
    sfmr_df[sfmr_lon] = sfmr_df[sfmr_lon] + pos_diff[lon]
    sfmr_df[sfmr_lat] = sfmr_df[sfmr_lat] + pos_diff[lat]

    err_df = sfmr_df[sfmr_df[sfmr_lon] > 180]
    if len(err_df.index > 0):
        raise ValueError("Longitude > 180")

    return sfmr_df


def clean_sfmr(sfmr_df, sfmr_source, sfmr_lon, sfmr_lat, sfmr_time, sfmr_quality, sfmr_wind):
    """
    Cleaning SFMR data from NaNs and from duplicates.


    Parameters
    ----------
    sfmr_df : pandas.DataFrame
              DataFrame containing SFMR data with datetime as index.
    sfmr_source : str
                  String indicating the SFMR data source.
    sfmr_lon : str
                Name of sfmr column to access longitude data.
    sfmr_lat : str
                Name of sfmr column to access latitude data.
    sfmr_time : str
                Name of sfmr time index.
    sfmr_quality : str
                   Name of sfmr column to access quality.
    sfmr_wind : str
                   Name of sfmr column to access wind speed.

    Returns
    -------
    pandas.DataFrame
        A pandas dataframe containing the cleaned data.
    """

    # Removing NaNs
    sfmr_df = sfmr_df[~pd.isna(sfmr_df[sfmr_lat])]
    sfmr_df = sfmr_df[~pd.isna(sfmr_df[sfmr_lon])]

    # Removing invalid data
    sfmr_df = sfmr_df[sfmr_df[sfmr_quality] <= 0]

    # HRD case : the time variable has to be formed from two other variables
    if sfmr_source == HRD:
        sfmr_df = sfmr_df[sfmr_df["DATE"] != 0]
        sfmr_df["DATE"] = sfmr_df["DATE"].astype(int).astype(str)
        sfmr_df["TIME"] = sfmr_df["TIME"].astype(int).astype(str)
        sfmr_df = sfmr_df[sfmr_df["DATE"].str.len() == 8]
        sfmr_df[sfmr_time] = sfmr_df.apply(lambda x: datetime.datetime.combine(
            datetime.datetime.strptime(x["DATE"], "%Y%m%d").date(),
            datetime.datetime.strptime(x["TIME"].zfill(6), "%H%M%S").time()
        ), axis=1)

    # Removing duplicates
    sfmr_df.drop_duplicates(subset=[sfmr_time, sfmr_wind], inplace=True)
    sfmr_df = sfmr_df.set_index(sfmr_time)

    return sfmr_df


def xr_rasterize(shapes, out_shape, transform, merge_alg=rio.enums.MergeAlg.replace):
    # rasterize, using merge_alg
    rasterized = rasterize(shapes, out_shape=out_shape, merge_alg=merge_alg, transform=transform, dtype='float64',
                           fill=0, default_value=0)

    # to xarray with lon/lat coords
    # dask.array.from_array(out)
    _, lat = transform * (0, np.arange(rasterized.shape[0]))
    lon, _ = transform * (np.arange(rasterized.shape[1]), 0)
    return xr.DataArray.from_dict(
        {
            'coords': {
                'lon': {'dims': 'lon', 'data': lon},
                'lat': {'dims': 'lat', 'data': lat},
            },
            'dims': ('lat', 'lon'),
            'data': rasterized,

        })


def get_transform(sat_ds, mission_config, lon, lat):
    if mission_config.pixel_spacing is None:
        if sat_ds.rio.transform() != Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0):
            transform = sat_ds.rio.transform() * Affine.translation(0.5, 0.5)
        else:
            logger.error("Couldn't find or compute a valid transform.")
            raise ValueError("Couldn't find or compute a valid transform.")
    else:
        lon_sign = int(np.sign(sat_ds.coords[lon][1] - sat_ds.coords[lon][0]))
        lat_sign = int(np.sign(sat_ds.coords[lat][1] - sat_ds.coords[lat][0]))
        transform = Affine(mission_config.pixel_spacing * lon_sign, 0.0, sat_ds.coords[lon][0].values,
                           0.0, mission_config.pixel_spacing * lat_sign, sat_ds.coords[lat][0].values)

    return transform


def sfmr_transform(sfmr_df, source_crs, target_crs, sfmr_lon, sfmr_lat):
    """
    Change sfmr lon/lat projection.

    Parameters
    ----------
    sfmr_df : pandas.DataFrame
              DataFrame containing SFMR data with datetime as index.
    source_crs : str
                 String representation of current CRS of SFMR data.
    source_crs : str
                 String representation of target CRS for SFMR data.
    sfmr_lon : str
                Name of sfmr column to access longitude data.
    sfmr_lat : str
                Name of sfmr column to access latitude data.

    Returns
    -------
    pandas.DataFrame
        SFMR dataframe with reprojected lon/lat.
    """

    gdf = gpd.GeoDataFrame(
        sfmr_df, geometry=gpd.points_from_xy(sfmr_df[sfmr_lon], sfmr_df[sfmr_lat]))
    gdf = gdf.set_crs(source_crs)
    gdf = gdf.to_crs(target_crs)

    gdf[sfmr_lon] = gdf["geometry"].apply(lambda x: x.x)
    gdf[sfmr_lat] = gdf["geometry"].apply(lambda x: x.y)

    df = pd.DataFrame(gdf)
    df = df.drop(columns=["geometry"])

    return df


def set_attributes(sfmr_file, sat_file, coloc_ds, sat_ds, mission_config, track_df, sfmr_time):
    """
    Set attributes to colocation xarray Dataset.

    Parameters
    ----------
    sfmr_file : str
                Path to currently processed SFMR file.
    sat_file : str
               Path to currently processed satellite file.
    coloc_ds : xarray.Dataset
               Dataset containing colocation data.
    sat_ds : xarray.Dataset
             Dataset of satellite data.
    mission_config : MissionConfig
                     MissionConfig instance for the current processed satellite.
    track_df : pd.DataFrame
               DataFrame containing cyclone track data.
    sfmr_time : str
                Name of Dataset variable to access SFMR time data.

    Returns
    -------
    xarray.Dataset
        The Dataset with added attributes.
    """

    mission = sat_ds.attrs["missionName"]
    coloc_ds.attrs["title"] = f"SFMR and {mission} colocation product."
    coloc_ds.attrs["institution"] = "IFREMER/LOPS"
    coloc_ds.attrs["mission_compared"] = mission
    coloc_ds.attrs["instrument_compared"] = mission_config.instrument
    coloc_ds.attrs["sat_footprint"] = sat_ds.attrs["footprint"]
    coloc_ds.attrs["cyclone_name"] = track_df.iloc[0]["name"]
    coloc_ds.attrs["cyclone_sid"] = track_df.iloc[0]["sid"]
    coloc_ds.attrs["satellite_acquisition_time"] = sat_ds.attrs[mission_config.acquisition_time_attribute]
    coloc_ds.attrs["filename_compared"] = os.path.basename(sat_file)
    coloc_ds.attrs["filename_sfmr"] = os.path.basename(sfmr_file)

    coloc_ds.coords["delta_time"].attrs = {
        "long_name": "Delta time",
        "description": "Delta time between satellite acquisition and SFMR acquisition points."
    }

    # coloc_ds["sfmr_translation_speed"].attrs = {
    #    "standard_name": "platform_speed_wrt_air",
    #    "units": "km/h"
    # }

    coloc_ds[sfmr_time].attrs = {
        "standard_name": "time",
        # "units": "seconds since UNIX epoch"
    }

    return coloc_ds


@jit(nopython=True, cache=True, nogil=True)
def numba_coloc(sfmr_time, sfmr_vars, sfmr_lon, sfmr_lat, sat_vars, sat_lon, sat_lat, dist_limit):
    """
    Associate each SFMR pixel with the closest SAR pixel.

    Parameters
    ----------
    sfmr_time : numpy.array of datetimes
                Array containing SFMR timestamps
    sfmr_vars : numba.typed.Dict(str: float64[:])
                Numba dict containing SFMR data (except time) as 1D numpy arrays of type float64
    sfmr_lon : str
                   Name of sfmr column to access longitude data
    sfmr_lat : str
                   Name of sfmr column to access latitude data
    sat_vars : numba.typed.Dict(str: float64[:])
                Numba dict containing satellite data (except time) as 1D numpy arrays of type float64
    sat_lon : numpy.array(dtype=float64)
                   Array containing satellite longitude grid
    sat_lat : numpy.array(dtype=float64)
                   Array containing satellite latitude grid
    dist_limit : float
                      Max spatial distance between SFMR and satellite pixel for the association to be valid.

    Returns
    -------
    numba.typed.Dict(str: float64[:])
        Numba dict containing associated satellite data
    """

    # Preparing arrays that will contain associated satellite data.
    sat_vars_coloc = dict()
    for var in sat_vars:
        sat_vars_coloc[var] = np.empty(sfmr_time.shape, dtype=types.float64)
        sat_vars_coloc[var][:] = np.nan

    count_skip = 0
    # For each sfmr point, finding the corresponding pixel in SAR grid
    for n in range(sfmr_time.shape[0]):
        dist = np.sqrt((sat_lon - sfmr_vars[sfmr_lon][n]) ** 2. +
                       (sat_lat - sfmr_vars[sfmr_lat][n]) ** 2.)

        min_dist = np.nanmin(dist)
        # If the distance is too large, we skip it
        if min_dist > dist_limit:
            count_skip += 1
            continue

        index = np.where(dist == min_dist)
        r_min = index[0][0]
        c_min = index[1][0]
        for var in sat_vars_coloc:
            sat_vars_coloc[var][n] = sat_vars[var][r_min, c_min]

    print("Skipped: ", count_skip)
    return sat_vars_coloc


def create_darray(data_dict, attrs, sfmr_time, delta_time):
    """
    Create xarray.DataArray using dict of {str: numpy arrays}.

    Parameters
    ----------
    data_dict : dict {str: numpy.array}
                Data dict to create DataArray from.
    attrs : dict {str: str}
            Dict of attributes to associate to each DataArray.
    sfmr_time : numpy.array
                Array of sfmr time data. Used for removing invalid data from other variables.
    delta_time : numpy.array
                 Coordinates array for each DataArray.

    Returns
    -------
    dict {str: xarray.DataArray}
        Python dict associating str with a DataArray.
    """
    ndarrays = {}

    # For every variable, building the DataArrays using the time dimension
    for varname in data_dict:
        da = data_dict[varname]
        # np.where(darray_count.values > 8, np.nan)
        da = da[~np.isnan(sfmr_time)]
        ndarrays[varname] = xr.DataArray.from_dict(
            {
                'coords': {
                    'delta_time': {'dims': 'delta_time', 'data': delta_time},
                },
                'dims': 'delta_time',
                'data': da,
                'attrs': attrs[varname]
            })

    return ndarrays


@timing
def fill_coloc(sfmr_df, sfmr_ds, sfmr_source, sfmr_time, sfmr_lon, sfmr_lat, sat_ds, sat_lon, sat_lat,
               mission_config, sat_acq_time, min_points):
    """
    Prepare colocation using numba and build final xarray.Dataset.

    Parameters
    ----------
    sfmr_df : pd.DataFrame
              DataFrame containing SFMR data with datetime as index.
    sfmr_ds : xarray.Dataset
              Source SFMR dataset used to retrieve attributes.
    sfmr_source : str
                  String indicating the SFMR data source.
    sfmr_time : str
                Name of sfmr time index.
    sfmr_lon : str
                Name of sfmr column to access longitude data.
    sfmr_lat : str
                Name of sfmr column to access latitude data.
    sat_ds : xarray.Dataset
             Source satellite Dataset.
    sat_lon : str
                Name of satellite longitude variable in Dataset.
    sat_lat : str
                Name of satellite latitude variable in Dataset.
    mission_config : MissionConfig
                     MissionConfig instance for the current processed satellite.
    sat_acq_time : datetime.datetime
                   Satellite acquisition time.
    min_points : int
                 Minimum number of points for the colocation to be valid. NOT IMPLEMENTED YET

    Returns
    -------
    xarray.Dataset
        Dataset containing colocated data.
    """

    sfmr_vars = Dict.empty(
        key_type=types.string,
        value_type=types.float64[:]
    )

    attrs = {}

    for var, mapped_var in sfmr_var_mapping[sfmr_source].items():
        if var != sfmr_time:
            sfmr_vars[mapped_var] = sfmr_df[var].to_numpy(dtype="float64")
        attrs[mapped_var] = sfmr_ds[var].attrs

    sfmr_final_lon = sfmr_var_mapping[sfmr_source][sfmr_lon]
    sfmr_final_lat = sfmr_var_mapping[sfmr_source][sfmr_lat]
    sfmr_final_time = sfmr_var_mapping[sfmr_source][sfmr_time]

    sat_vars = Dict.empty(
        key_type=types.string,
        value_type=types.float64[:, :]
    )

    # Satellites variables previous:new names mapping.
    # Each entry will result in a variable added in the final Dataset.
    varname_sat_mapping = mission_config.var_mapping
    for var, mapped_var in varname_sat_mapping.items():
        if var in sat_ds:
            sat_vars[mapped_var] = sat_ds[var].values.squeeze().astype(dtype="float64")
            attrs[mapped_var] = sat_ds[var].attrs

    sat_lon_array = sat_ds.coords[sat_lon].values.flatten().astype(dtype="float64")
    sat_lat_array = sat_ds.coords[sat_lat].values.flatten().astype(dtype="float64")

    sat_lon_array, sat_lat_array = np.meshgrid(sat_lon_array, sat_lat_array)

    sfmr_time_array = sfmr_df[sfmr_time].to_numpy()
    sat_vars_coloc = numba_coloc(sfmr_time_array, sfmr_vars, sfmr_final_lon, sfmr_final_lat, sat_vars, sat_lon_array,
                                 sat_lat_array, mission_config.max_coloc_dist)

    sat_acq_time = pd.to_datetime(sat_acq_time, utc=False).tz_localize(None)
    sfmr_vars = dict(sfmr_vars)
    sfmr_vars[sfmr_final_time] = sfmr_time_array.astype("datetime64[s]")
    delta_time = sfmr_df[sfmr_time] - sat_acq_time

    ndarrays = create_darray(sfmr_vars, attrs, sfmr_time_array, delta_time)
    ndarrays.update(create_darray(sat_vars_coloc, attrs, sfmr_time_array, delta_time))

    # Creating the Dataset
    coloc_ds = xr.Dataset(data_vars=ndarrays)
    coloc_ds = coloc_ds.sortby('delta_time')

    p1 = pyproj.Proj(sat_ds.rio.crs)
    p2 = pyproj.Proj("EPSG:4326")
    coloc_ds["sfmr_lon"].values, coloc_ds["sfmr_lat"].values = pyproj.transform(p1, p2, coloc_ds["sfmr_lon"].values,
                                                                                coloc_ds["sfmr_lat"].values,
                                                                                always_xy=True)

    return coloc_ds


def write_coloc(output_path, coloc_ds, sfmr_file, sat_mission, acq_time, sfmr_source):
    """
    Write colocation Dataset to netCDF file using the following format :
        <SFMR_source>__<sat_mission>_<timestamp>.nc

    Parameters
    ----------
    output_path : str
                 Path in which the file will be written.
    coloc_ds : xarray.Dataset
               Dataset of colocated data. This is the dataset that is written to file.
    sat_mission : str
                  String representing satellite mission.
    acq_time : datetime.datetime
               Datetime of satellite acquisition.
    sfmr_source : str
                  String representing SFMR data source.

    """
    # sfmr_id = os.path.splitext(os.path.basename(sfmr_file))[0]
    s = os.path.splitext(os.path.basename(sfmr_file))[0]
    filename = f"{s}_{sat_mission}_{acq_time.strftime('%Y%m%dT%H%M%S')}.nc"

    out_filepath = os.path.join(output_path, filename)
    logger.info(f"Writing to {out_filepath}")

    coloc_ds.to_netcdf(out_filepath)


# @jit(nopython=True)
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance and bearing between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    bearing = np.arctan2(np.sin(lon2 - lon1) * np.cos(lat2),
                         np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))
    return c * r, np.rad2deg(bearing)


@timing
def sfmr_translation_speed(sfmr_df, sfmr_lon, sfmr_lat):
    """
    Add a column names "translation speed" to the given dataframe, which contains the sfmr plane translation speed
    computed from longitude, latitude and time columns.

    Parameters
    ----------
    sfmr_df : pandas.DataFrame
        The pandas dataframe containing SFMR data. Needs longitude latitude as columns and time as index.

    Returns
    -------
    pandas.DataFrame
        Returns the same dataframe with translation_speed column added
    """

    distance, _ = haversine(sfmr_df[sfmr_lat].shift(1), sfmr_df[sfmr_lon].shift(1),
                            sfmr_df.iloc[1:][sfmr_lat], sfmr_df.iloc[1:][sfmr_lon])

    time_diff = sfmr_df.index.to_series().diff()
    time_diff = time_diff.apply(lambda x: x.total_seconds() / 60 / 60)
    speed = distance / time_diff
    sfmr_df["translation_speed"] = speed

    return sfmr_df


def generate_coloc(sat_file, sat_mission, sfmr_file, track_df, mission_config, out_path, min_points):
    """
    Main function to generate colocation for a SFMR and a satellite file.
    The are opened, cleaned, formatted and then colocated. The generated colocation is then written to disk.

    Parameters
    ----------
    sat_file : str
               Path to satellite NETCDF file.
    sat_mission : str
                  Satellite mission.
    sfmr_file : str
                Path to SFMR NETCDF file.
    track_df : pandas.DataFrame
               Dataframe containg track data about the cyclone concerned by the satellite and SFMR acquisition.
    mission_config : MissionConfig
                     MissionConfig instance for the current satellite file.
    out_path : str
               Output path in which the file will be written.
    """
    logger.info(f"Generating coloc for {sat_file}, {sfmr_file}")

    # Open SFMR track file with xarray
    sfmr_ds = xr.open_dataset(sfmr_file)

    sfmr_source = sfmr_data_source(sfmr_ds)

    # SFMR column names to use
    sfmr_lon = sfmr_sources_variables[sfmr_source]["lon"]
    sfmr_lat = sfmr_sources_variables[sfmr_source]["lat"]
    sfmr_time = sfmr_sources_variables[sfmr_source]["time"]
    sfmr_wind = sfmr_sources_variables[sfmr_source]["wind_speed"]
    sfmr_quality = sfmr_sources_variables[sfmr_source]["quality"]

    # Open satellite file with xarray
    sat_ds = xr.open_dataset(sat_file)

    sat_ds = mission_config.prepare_vars(sat_ds)

    # Lon, lat coord names for satellite data
    sat_lon = get_lon(sat_ds)
    sat_lat = get_lat(sat_ds)

    # Converting to pandas dataframe
    sfmr_df = sfmr_ds.to_dataframe()

    # Cleaning the dataframe to be sure we won't stumble on a NaN
    sfmr_df = clean_sfmr(sfmr_df, sfmr_lon=sfmr_lon, sfmr_lat=sfmr_lat, sfmr_time=sfmr_time,
                         sfmr_wind=sfmr_wind, sfmr_quality=sfmr_quality, sfmr_source=sfmr_source)

    # Compute sfmr plane speed and add it to dataframe
    # sfmr_df = sfmr_translation_speed(sfmr_df, sfmr_lon=sfmr_lon, sfmr_lat=sfmr_lat)

    # Cleaning satellite data
    sat_ds = mission_config.clean_data(sat_ds)

    excluded_resample = [sfmr_var_mapping_reversed[sfmr_source][v] for v in sfmr_time_resample_excluded]
    # Temporally downsample SFMR data
    sfmr_df = sfmr_resample(sfmr_df, mission_config.sfmr_resample_freq_sec, sfmr_lon, excluded_resample)

    # Spatially downsample SFMR data
    sfmr_df = spatial_downsample(sfmr_df, sfmr_lat, sfmr_lon, sfmr_time, mission_config.sfmr_spatial_downsample_dist, sfmr_source)

    acq_time = get_sat_acq_time(sat_ds, mission_config)
    sfmr_df_trans = sfmr_translate(sfmr_df, track_df, acq_time, sfmr_lon, sfmr_lat, sfmr_time)

    sfmr_df_trans = sfmr_transform(sfmr_df_trans, source_crs="EPSG:4326", target_crs=sat_ds.rio.crs,
                                   sfmr_lon=sfmr_lon, sfmr_lat=sfmr_lat)

    # coloc_ds = fill_coloc_df(sfmr_df_trans, sfmr_ds, sat_ds, acq_time,
    #                         mission_config, min_points, sat_file, sfmr_file,
    #                         lon, lat, sfmr_lon, sfmr_lat, sfmr_time, sfmr_source)

    # Compute the concrete colocation : associating each SFMR pixel with a satellite pixel.
    coloc_ds = fill_coloc(sfmr_df_trans, sfmr_ds, sfmr_source, sfmr_time, sfmr_lon,
                          sfmr_lat, sat_ds, sat_lon, sat_lat, mission_config, acq_time, min_points)

    # If the colocation is valid
    if coloc_ds is not None:
        coloc_ds = set_attributes(sfmr_file, sat_file, coloc_ds, sat_ds, mission_config, track_df,
                                  sfmr_var_mapping[sfmr_source][sfmr_time])

        write_coloc(out_path, coloc_ds, sfmr_file, sat_mission, acq_time, sfmr_source)


def prepare_sfmr(sfmr_file, track_df, acq_time):
    """
    Custom function to do what I need with the SFMR.
    Replaces the function generate_coloc() used by Theo on Cyclobs.
    
    Parameters
    ----------
    sfmr_file : str
                Path to SFMR NETCDF file.
    track_df : pandas.DataFrame
               Dataframe containg track data about the cyclone concerned by the satellite and SFMR acquisition.
    acq_time : datetime.datetime
               Satellite acquisition datetime.
    """

    logger.info(f"{sfmr_file}")

    # Open SFMR track file with xarray
    sfmr_ds     = xr.open_dataset(sfmr_file)
    # HRD or NESDIS
    sfmr_source = sfmr_data_source(sfmr_ds)

    # SFMR column names to use
    sfmr_lon     = sfmr_sources_variables[sfmr_source]["lon"]
    sfmr_lat     = sfmr_sources_variables[sfmr_source]["lat"]
    sfmr_time    = sfmr_sources_variables[sfmr_source]["time"]
    sfmr_wind    = sfmr_sources_variables[sfmr_source]["wind_speed"]
    sfmr_quality = sfmr_sources_variables[sfmr_source]["quality"]

    # Converting to pandas dataframe
    sfmr_df = sfmr_ds.to_dataframe()

    # Cleaning the dataframe to be sure we won't stumble on a NaN
    sfmr_df = clean_sfmr(sfmr_df, sfmr_lon=sfmr_lon, sfmr_lat=sfmr_lat, sfmr_time=sfmr_time,
                         sfmr_wind=sfmr_wind, sfmr_quality=sfmr_quality, sfmr_source=sfmr_source)

    excluded_resample = [sfmr_var_mapping_reversed[sfmr_source][v] for v in sfmr_time_resample_excluded]
    # Temporally downsample SFMR data
    sfmr_resample_freq_sec = 10
    sfmr_df                = sfmr_resample(sfmr_df, sfmr_resample_freq_sec, sfmr_lon, excluded_resample)

    # Spatially downsample SFMR data
    sfmr_downsample_dist = 0.03
    sfmr_df              = spatial_downsample(sfmr_df, sfmr_lat, sfmr_lon, sfmr_time, sfmr_downsample_dist, sfmr_source)

    sfmr_df_trans = sfmr_translate(sfmr_df, track_df, acq_time, sfmr_lon, sfmr_lat, sfmr_time)
    

    # A voir si tu en as besoin, c'est pour changer SFMR de projection spatiale
    # sfmr_df_trans = sfmr_transform(sfmr_df_trans, source_crs="EPSG:4326", target_crs=sat_ds.rio.crs,
    #                               sfmr_lon=sfmr_lon, sfmr_lat=sfmr_lat)

    return sfmr_df


@timing
def generate_all_coloc(out_path, host_api, days_to_update=None, min_points=50, sat_files=None, input_dir=None,
                       llgd=False, sfmr_source="all", sat_mission="all", sat_instrument="all", sar_commit=None,
                       sar_config=None):
    """
    Finds SFMR-acquisition-Satellite_acquisition-Cyclone association to use for colocations. Those associations
    are found using Cyclobs API. The association used for colocation can be filtered with available parameters.

    Parameters
    ----------
    out_path : str
               Path in which colocation products will be written.
    host_api : str
               URL of the host API.
    days_to_update : int
                     Number of days to update before now.
    min_points : int
                 Minimum number of points for the colocation to be valid.
    sat_files : list(str)
                list of satellites files to filter SFMR-SAT-cyclone associations.
    ll_gd : bool
            Use SAR ll_gd files. Use _gd files if False.
    sfmr_source : str
                  SFMR data source to use. {"HRD", "NESDIS", "all"}
    sat_mission : str
                  Sat mission to filter on for finding associations.
    sat_instrument : str
                  Sat instrument to filter on for finding associations.
    sar_commit : str
                Override sarwing default commit. Will modify files found through CyclObs API to set the given commit
    sar_config : str
                Override sarwing default config. Will modify files found through CyclObs API to set the given config
    """

    if days_to_update is not None:
        min_date = (datetime.datetime.now() - datetime.timedelta(days=days_to_update)).strftime("%Y-%m-%d")
    else:
        min_date = "1970-01-01"

    if sfmr_source == "all":
        sources = ["NESDIS", "HRD"]
        allowed_source = "noaa_aoml_hrd|noaa_nesdis_star"
    elif sfmr_source == "HRD":
        sources = ["HRD"]
        allowed_source = "noaa_aoml_hrd"
    elif sfmr_source == "NESDIS":
        sources = ["NESDIS"]
        allowed_source = "noaa_nesdis_star"
    else:
        msg = f"Unknown source {sfmr_source}"
        raise ValueError(msg)

    if input_dir is not None:
        if llgd:
            input_files = glob.glob(os.path.join(input_dir, "*_ll_gd.nc"))
        else:
            input_files = glob.glob(os.path.join(input_dir, "*_gd.nc"))

        input_base_files = [os.path.basename(f) for f in input_files]

        if len(input_files) == 0:
            msg = f"No .nc files found from {input_dir}"
            raise FileNotFoundError(msg)

    for src in sources:
        logger.info(f"Processing for source : {src}...")
        api_req = f"{host_api}/app/api/getData?include_cols=all&track_source=sfmr_{src}&acquisition_start_time={min_date}"

        if sat_files is not None:
            api_req += "&filename=" + ",".join(sat_files)

        if sat_mission != "all":
            api_req += f"&mission={sat_mission}"

        if sat_instrument != "all":
            api_req += f"&instrument={sat_instrument}"

        df = pd.read_csv(api_req)
        df["data_url"] = df["data_url"].apply(lambda x: pathurl.toPath(x, schema="dmz"))
        df = df[df["track_file"].str.contains(allowed_source)]

        path_mod_instruments = ["C-Band SAR"]
        # Modify SAR config and/or commit in path if requested. This is to be able to use data from
        # wanted sarwing listing.
        if sar_commit is not None:
            df.loc[df["instrument_short"].isin(path_mod_instruments), "data_url"] = \
                df.loc[df["instrument_short"].isin(path_mod_instruments), "data_url"].apply(
                    lambda x: x.replace(x.split("/")[9], sar_commit))
        if sar_config is not None:
            df.loc[df["instrument_short"].isin(path_mod_instruments), "data_url"] = \
                df.loc[df["instrument_short"].isin(path_mod_instruments), "data_url"].apply(
                    lambda x: x.replace(x.split("/")[10], sar_config))

        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_colwidth', 1000)

        # Here it is possible to apply a criteria on the number of colocated Track point.
        # df.groupby(["data_url"]).filter(lambda x: len(x) > 10)
        # Keeping only one set of each association of satellite acquisition file and cyclone id
        # df = df.drop_duplicates(subset=["data_url", "sid", "track_file"])

        # Process each file association
        for index, row in df.iterrows():
            data_url = row["data_url"]

            # If files comes from an input directory, verify that the file exist, if it does, use it. Else, skip it.
            if input_dir is not None:
                rep = "_ll_gd" if llgd else "_gd"
                check_for = os.path.basename(row["data_url"]).replace("_ll_gd", rep)
                print(check_for)
                print(input_base_files[0])
                if check_for not in input_base_files:
                    continue
                else:
                    data_url = os.path.join(input_dir, check_for)

            # Matches the sid atcf format. Examples : al062018 or wp192020 etc
            # The condition prevents from generate a coloc when a SFMR track has not been associated to an atcf track
            # in the database. (When there's no association in the DB, the SFMR Track sid field is filled
            # with the sfmr filename.
            if re.match(r"[a-z]{2}[0-9]{6}", row["sid"]) and "SFMR" not in row["sid"] and "sfmr" not in row["sid"]:
                coloc_from_inputs(sfmr_file=row["track_file"], sat_file=data_url,
                                  sid=row["sid"], mission_short=row["mission_short"], out_path=out_path,
                                  host_api=host_api,
                                  min_points=min_points, llgd=llgd)


def coloc_from_inputs(sfmr_file, sat_file, sid, mission_short, out_path, host_api, min_points, llgd=False):
    """
    Finds associated Cyclones track for a SFMR-Satellite association using Cyclobs API, then starts colocation
    generation.

    Parameters
    ----------
    sfmr_file : str
                SFMR file to use for colocation.
    sat_file : str
               Satellite file to use for colocation.
    sid : str
          Concerned cyclone sid.
    mission_short : str
                    Short version of current satellite mission.
    out_path : str
               Path to which the colocation product will be written.
    host_api : str
               Host to use to query API.
    min_points : int
                 Minimum number of points for colocation to be valid. Not implemented yet.
    ll_gd : bool
            Use SAR ll_gd files. Use _gd files if False.
    """

    track_req = f"{host_api}/app/api/track?include_cols=all&source=atcf&freq=1&sid={sid}"
    track_df = pd.read_csv(track_req)

    track_df["datetime"] = pd.to_datetime(track_df["date"], format="%Y-%m-%d %H:%M:%S")

    mission = mission_short
    if mission in SAR_missions:
        mission_config = SARConfig()
        if not llgd:
            sat_file = sat_file.replace("_ll", "")
    elif mission in l_band_missions:
        mission_config = LBandConfig()
    else:
        err_msg = f"Unable to find configuration for satellite mission {mission}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    generate_coloc(sat_file, mission_short, sfmr_file, track_df, mission_config, out_path, min_points)


### OLD FUNC

@timing
def fill_coloc_df(sfmr_df_trans, sfmr_ds, sat_ds, sat_acq_time,
                  mission_config, min_points, sat_file, sfmr_file,
                  lon, lat, sfmr_lon, sfmr_lat, sfmr_time, sfmr_source):
    sgdf = gpd.GeoDataFrame(
        sfmr_df_trans, geometry=gpd.points_from_xy(sfmr_df_trans[sfmr_lon], sfmr_df_trans[sfmr_lat]))

    # Transform time to POSIX format (seconds since UNIX time) because rasterize doesn't work with datetime type.
    sgdf["timestamp"] = sgdf[sfmr_time].apply(lambda x: x.timestamp())

    transform = get_transform(sat_ds, mission_config, lon, lat)

    # Getting points from sfmr data
    shapes = ((geom, 1) for geom, value in zip(sgdf.geometry, sgdf.timestamp))
    # Burning the SFMR points on the satellite data. For each SFMR point, the value "1" is associated.
    # When a point is burned into the data array, it'll add "1" to the current value. It makes
    # possible to count how many SFMR points are associated to each satellite pixel.
    # darray_count is the array containing how many SFMR pixels are associated to each satellite pixel
    darray_count = xr_rasterize(shapes, out_shape=sat_ds["wind_speed"].squeeze().shape,
                                merge_alg=rio.enums.MergeAlg.add,
                                transform=transform)

    # SFMR variables previous:new names mapping.
    # Each entry will result in a variable added in the final Dataset.
    varname_mapping = sfmr_var_mapping[sfmr_source]
    darrays = {}
    # For each variable
    for colname in sgdf.columns.tolist():
        if colname in varname_mapping:
            # Burning each variable value into the satellite data.
            # If multiple values end up in the same satellite pixel, they'll be added. Because of this,
            # we later use the darray_count to do a mean.
            shapes = ((geom, value) for geom, value in zip(sgdf.geometry, sgdf[colname]))
            darray = xr_rasterize(shapes, out_shape=sat_ds["wind_speed"].squeeze().shape,
                                  merge_alg=rio.enums.MergeAlg.add,
                                  transform=transform)
            # Here is the mean operation
            darray_mean = darray / darray_count
            darrays[varname_mapping[colname]] = darray_mean
            if colname == "translation_speed" or colname == "timestamp":
                darrays[varname_mapping[colname]].attrs = {}
            else:
                darrays[varname_mapping[colname]].attrs = sfmr_ds[colname].attrs
                # darrays[varname_mapping[colname]].attrs["standard_name"] = \
                #    "sfmr_" + darrays[varname_mapping[colname]].attrs["standard_name"]

    # Satellites variables previous:new names mapping.
    # Each entry will result in a variable added in the final Dataset.
    varname_sat_mapping = mission_config.var_mapping
    for varname, da in sat_ds.data_vars.items():
        if varname in varname_sat_mapping:
            darrays[varname_sat_mapping[varname]] = da

    # Preparing the time variable for computing delta time
    time_sfmr = darrays["sfmr_time"].values.flatten()
    time_var = darrays["sfmr_time"].values.flatten()
    time_var = time_var[~np.isnan(time_var)]
    time_df = pd.to_datetime(time_var.astype(int), unit="s", utc=True)

    # Computing time delta variable (time between satellite acquisition and SFMR acquisition)
    time_df = time_df.to_series()

    # Need to convert like because of a bug : https://github.com/pandas-dev/pandas/issues/32619
    sat_acq_time = pd.to_datetime(sat_acq_time, utc=True)
    delta_time = time_df - pd.to_datetime(sat_acq_time)

    # darrays["sfmr_time"] = darrays["sfmr_time"].astype('datetime64[s]')
    # darrays["sfmr_time"].values[:] = pd.to_datetime(darrays["sfmr_time"].values.astype(int).astype(str), unit="s", utc=True)
    # datetime.datetime.utcfromtimestamp(darrays["sfmr_time"].values.astype(int))
    # sat_ds = sat_ds.assign(sfmr_wind_speed=darrays["sfmr_wind_speed"])
    # sat_ds = sat_ds.assign(sfmr_time=darrays["sfmr_time"])
    # sat_ds.to_netcdf("./temp.nc")

    darray_count = darray_count.values.flatten()

    count_no_zero = darray_count[darray_count > 0]
    count_limit = np.nanmean(count_no_zero) + 2 * np.nanstd(count_no_zero)

    print("LIMIT", count_limit)
    import copy
    d_count = copy.copy(darray_count)
    dc = darray_count[darray_count >= count_limit]
    d_coloc = darray_count[darray_count > 1]
    print("Removed count : ", dc.shape, d_count.shape, d_coloc.shape)

    ndarrays = {}
    # For every variable, building the DataArrays using the time dimension
    for varname in darrays:
        da = darrays[varname].values
        da = da.flatten()
        # np.where(darray_count.values > 8, np.nan)
        da = np.where(darray_count > count_limit, np.nan, da)
        da = da[~np.isnan(time_sfmr)]
        print(da.shape)
        ndarrays[varname] = xr.DataArray.from_dict(
            {
                'coords': {
                    'delta_time': {'dims': 'delta_time', 'data': delta_time},
                },
                'dims': 'delta_time',
                'data': da,
                'attrs': darrays[varname].attrs
            })

    wdspd = ndarrays["sat_wind_speed"].values
    nb_coloc_points = wdspd[~np.isnan(wdspd)].size
    logger.info(f"Number of colocated points : {nb_coloc_points}")
    if nb_coloc_points < min_points:
        logger.warning(f"Colocation for satellite product ({sat_file}) and SFMR product ({sfmr_file}) failed because "
                       f"only {nb_coloc_points} are colocated. At least {min_points} are required for the colocation "
                       f"to be valid.")
        return None

    darray_count = np.where(darray_count > count_limit, np.nan, darray_count)
    darray_count = darray_count[~np.isnan(time_sfmr)]
    ndarrays["count"] = xr.DataArray.from_dict({
        'coords': {
            'delta_time': {'dims': 'delta_time', 'data': delta_time},
        },
        'dims': 'delta_time',
        'data': darray_count,
    })

    # Creating the Dataset
    coloc_ds = xr.Dataset(data_vars=ndarrays)
    coloc_ds = coloc_ds.sortby('delta_time')

    p1 = pyproj.Proj(sat_ds.rio.crs)
    p2 = pyproj.Proj("EPSG:4326")
    coloc_ds["sfmr_lon"].values, coloc_ds["sfmr_lon"].values = pyproj.transform(p1, p2, coloc_ds["sfmr_lon"].values,
                                                                                coloc_ds["sfmr_lon"].values)

    return coloc_ds
