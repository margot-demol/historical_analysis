"""
LIBRARY
-------
"""
import numpy as np
import pandas as pd
import xarray as xr
import pyproj
from rasterio.transform import Affine

import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.geodesic as cgeo

crs = ccrs.PlateCarree()
import cmocean.cm as cm

from xgcm import Grid
from xhistogram.xarray import histogram
import warnings

warnings.filterwarnings("ignore")

import os
from glob import glob

"""
READING DATA
---------------------------------------------------------------------------------------------------------
"""

colloc_path = {
    2008: "/home/datawork-cersat-public/cache/users/jfpiolle/felyx/drifters/mdb_v2/2008/",
    2009: "/home/datawork-cersat-public/cache/users/jfpiolle/felyx/drifters/mdb_v2/2009/",
    2010: "/home/datawork-cersat-public/cache/users/jfpiolle/felyx/drifters/mdb_v2/2010/",
    2011: "/home/datawork-cersat-public/cache/users/jfpiolle/felyx/drifters/mdb_v2/2011/",
    2012: "/home/datawork-cersat-public/cache/users/jfpiolle/felyx/drifters/mdb_v2/2012/",
    2013: "/home/datawork-cersat-public/cache/users/jfpiolle/felyx/drifters/mdb_v2/2013/",
    2014: "/home/datawork-cersat-public/cache/users/jfpiolle/felyx/drifters/mdb_v2/2014/",
    2015: "/home/datawork-cersat-public/cache/users/jfpiolle/felyx/drifters/mdb_v2/2015/",
    2016: "/home/datawork-cersat-public/cache/users/jfpiolle/felyx/drifters/mdb_v2/2016/",
    2017: "/home/datawork-cersat-public/cache/users/jfpiolle/felyx/drifters/mdb_v2/2017/",
    2018: "/home/datawork-cersat-public/cache/users/jfpiolle/felyx/drifters/mdb_v2/2018/",
    2019: "/home/datawork-cersat-public/cache/users/jfpiolle/felyx/drifters/mdb_v2/2019/",
    2020: "/home/datawork-cersat-public/cache/users/jfpiolle/felyx/drifters/mdb_v2/2020/",
}


def load_collocalisations(source, satellite=None, product_type=None, drifter=None):
    """Load collocalisations altimetry / GDP data

    Parameters
    ----------
    source : str, int
        Data source, e.g. 2016, 2018
    satellite : str
        type of satellite we want to select among "SARAL" or "Sentinel"  (if we want to)
    drifter : str
        type of drifter we want to select among "gps" or "argos" (if we want to)

    Return
    ------
    list of .nc filess

    """
    files = sorted(glob(os.path.join(colloc_path[source], "*.nc")))

    if product_type is not None:
        files = [f for f in files if satellite in f]
    if satellite is not None:
        files = [f for f in files if product_type in f]
    if drifter is not None:
        files = [f for f in files if drifter in f]
    files = [f for f in files if xr.open_dataset(f).dims['obs']>1] # 1 obs files can't be processed by build_dataset (TO CORRECT ONE DAY...)
    
    return files


def change_prefix(ds):
    """Find the data prefixes

    Parameters
    ----------
    ds : xr.DataArray
        colocalisations dataset

    Return
    ------
    drifters prefix, satellite prefix : str, str

    """
    argos = any(["argos" in variable for variable in list(ds)])
    sla = any(["sea_level_anomaly_lrrmc" in variable for variable in list(ds)])

    if argos:
        ds = ds.rename({v: v.replace("argos_drifters", "drifter") for v in ds})
    else:
        ds = ds.rename({v: v.replace("gps_drifters", "drifter") for v in ds})
    if sla:
        ds = ds.rename({v: v.replace("sea_level_anomaly_lrrmc", "sla") for v in ds})

    ds = ds.rename({v: v.replace("cmems", "alti") for v in ds})
    ds = ds.rename_dims({"cmems_time": "alti_time"})

    return ds


"""
BOX
---------------------------------------------------------------------------------------------------------
"""


def get_proj(lonc, latc):  # converts from lon, lat to native map projection x,y
    """Create pyproj Proj object, project is an azimutal Eqsuidistant projection centered on the central point of the selected satellite track = matching point
    https://proj.org/operations/projections/aeqd.html

    Parameters
    ----------
    lonc,latc : float
        central longitude and latitude of the satellite track, matching point on which the box will be centered
    Return
    ------
    pyproj.Proj object
    """
    return pyproj.Proj(
        proj="aeqd", lat_0=latc, lon_0=lonc, datum="WGS84", units="m"
    )  # aeqd Azimutal EQuiDistant projection centered on lonc,latc


def compute_box_orientation(lonc, latc, lon0, lat0, lon1, lat1):
    """Compute the orientation of the box i.e. the angle (-180 and 180°) between the oriented track (time=0 -> end) and the longitude axe.

    Parameters
    ----------
    lonc,latc : float
        central longitude and latitude of the satellite track, matching point on which the box will be centered
    lon1,lat1 : float
        longitude and latitude of the end of the satellite track
    Return
    ------
    box orientation
    """
    proj = get_proj(lonc, latc)
    # get local coordinate
    x0, y0 = proj.transform(lon0, lat0)  # xc=yc=0 origin of the box grid
    x1, y1 = proj.transform(lon1, lat1)
    # get orientation of defined by central point and point 1

    phi = (
        np.arctan2(y1 - y0, x1 - x0) * 180 / np.pi
    )  # angle (-180 and 180°) between the oriented track (time=0 -> end) and the longitude axe
    # assert  phi==0, (x0,y0, x1,y1)
    return phi


# compute_box_vec = np.vectorize(compute_box_core)


def compute_box(ds):
    """Compute box properties around each collocation:
    central position and orientation

    Parameters
    ----------
    ds : xr.DataArray
        colocalisations dataset
    asuff : str
        satellite prefixe

    Return
    ------
    central longitude, central latitude, orientation : xr.Datarray, xr.Datarray, xr.Datarray
    """
    i = (ds["alti_time"].size - 1) // 2
    lonc = ds["alti_lon"].isel(alti_time=i)
    latc = ds["alti_lat"].isel(alti_time=i)
    # lon1 = ds["alti_lon"].isel(alti_time=-1)
    # lat1 = ds["alti_lat"].isel(alti_time=-1)
    lon = ds["alti_lon"].ffill(dim="alti_time").bfill(dim="alti_time")
    lat = ds["alti_lat"].ffill(dim="alti_time").bfill(dim="alti_time")
    lon1 = lon.isel(alti_time=-1)
    lat1 = lat.isel(alti_time=-1)
    lon0 = lon.isel(alti_time=0)
    lat0 = lat.isel(alti_time=0)
    # assert lonc.where(lonc==1., drop=True).size==0, 'dataarray lonc=1'

    # will need to vectorize with dask eventually
    phi = xr.apply_ufunc(
        compute_box_orientation,
        lonc,
        latc,
        lon0,
        lat0,
        lon1,
        lat1,
        # input_core_dims = [[]]*4,
        vectorize=True,
        dask="parallelized",
    ).rename("phi")

    # phi = compute_box_vec(lonc, latc, lon1, lat1)

    return lonc, latc, phi


""" 
BOX X,Y -> LON, LAT
--------------------------------------------
"""


def compute_box_grid_lonlat_core(lonc, latc, phi, x=None, y=None):
    """Compute coordinates longitude, latitude of the local box grid (x-along satellite track, y-normal to satellite track -> lon,lat)
    https://github.com/rasterio/affine

    Parameters
    ----------
    lonc, latc, phi: float  central position and orientation of the box
    x, y: np.array local grid of the box (with origin at (lonc, latc) and x-axis aligned
    with the satellite track (lonc, latc) - (lon1, lat1) direction)

    Return
    ------
    longitude, latitude : np.array, np.array

    """
    xv, yv = np.meshgrid(x, y)
    proj = get_proj(lonc, latc)
    # assert False, help(proj.transform)
    xc, yc = proj.transform(lonc, latc)
    # apply inverse affine transformation
    a_fwrd = Affine.translation(-xc, -yc) * Affine.rotation(-phi, pivot=(xc, yc))
    a_back = ~a_fwrd
    # compute coordinates of x,y in the lon, lat orientated grid
    xv_inv, yv_inv = a_back * (xv, yv)
    lon, lat = proj.transform(
        xv_inv,
        yv_inv,
        direction=pyproj.enums.TransformDirection.INVERSE,
    )
    return lon, lat


def compute_box_grid_lonlat(ds, x, y):
    """Compute local coordinates in longitude, latitude of the local box grid (x-along satellite track, y-normal to satellite track -> lon,lat) for all colocalisations

     Parameters
    ----------
    ds : xr.DataArray
        colocalisations dataset
    x, y: np.array
        local grid of the box (with origin at (lonc, latc) and x-axis aligned )

    Return
    ------
    longitude, latitude : xr.Datarray, xr.Datarray

    """
    lon, lat = xr.apply_ufunc(
        compute_box_grid_lonlat_core,
        ds.box_lonc,
        ds.box_latc,
        ds.box_phi,
        kwargs=dict(x=x, y=y),
        output_core_dims=[["box_y", "box_x"]] * 2,
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs=dict(output_sizes=dict(box_x=x.size, box_y=y.size)),
    )
    lon = lon.assign_coords(box_x=x, box_y=y)
    lat = lat.assign_coords(box_x=x, box_y=y)
    return lon, lat


""" 
LON, LAT -> X,Y
--------------------------------------------
"""


def lonlat2xy(lonc, latc, phi, lon, lat, lon1=None, lat1=None):
    """return coordinates with origin at (lonc, latc) and x-axis aligned
    with (lonc, latc) - (lon1, lat1) direction (lon,lat -> x-along satellite track, y-normal to satellite track)

    Parameters
    ----------
    lonc, latc, phi: float
        central position and orientation of the box
    lon, lat : np.array, np.array
        local grid of the box
    lon1,lat1 : float
        end of the satellite track

    Return
    ------
    local x, local y : np.array, np.array

    """
    proj = get_proj(lonc, latc)
    # get local coordinate
    xc, yc = proj.transform(lonc, latc)
    xl, yl = proj.transform(lon, lat)
    if phi is None:
        # compute phi from x1 and y1
        x1, y1 = proj.transform(lon1, lat1)
        # get orientation of defined by central point and point 1
        phi = np.arctan2(y1 - yc, x1 - xc) * 180 / np.pi
    # build affine operators
    a_fwrd = Affine.translation(-xc, -yc) * Affine.rotation(-phi, pivot=(xc, yc))
    # a_back = ~a_fwrd

    x, y = a_fwrd * (xl, yl)

    return x, y


def compute_local_xy(ds, prefixe, tdim):
    """return coordinates with origin at (lonc, latc) and x-axis aligned
    with (lonc, latc) - (lon1, lat1) direction (lon,lat -> x-along satellite track, y-normal to satellite track) for all colocalisations

    Parameters
    ----------
    ds: xr.DataArray
        colocalisations dataset
    datatype: 'drifter' or 'alti'

    Return
    ------
    local x, local y : xr.Datarray, xr.Datarray

    """

    lon = ds[prefixe + "_lon"]
    lat = ds[prefixe + "_lat"]

    # will need to vectorize with dask eventually
    x, y = xr.apply_ufunc(
        lonlat2xy,
        ds.box_lonc,
        ds.box_latc,
        ds.box_phi,
        lon,
        lat,
        input_core_dims=[[]] * 3 + [[tdim]] * 2,
        output_core_dims=[[tdim]] * 2,
        vectorize=True,
        dask="parallelized",
    )
    return x, y


""" 
THETAS
--------------------------------------------
"""


def compute_box_thetas(ds):
    """Compute the local angles theta between e_lon and x and between e_lat and x for the box of one colocalisation
    https://github.com/rasterio/affine

    Parameters
    ----------
    ds : xr.DataArray
        colocalisations dataset
    Return
    ------
    box_theta_longitude, box_theta_latitude : np.array, np.array


    """

    dlon_dx, dlon_dy = ds.box_lon.differentiate("box_x"), ds.box_lon.differentiate(
        "box_y"
    )
    dlat_dx, dlat_dy = ds.box_lat.differentiate("box_x"), ds.box_lat.differentiate(
        "box_y"
    )

    return np.arctan2(-dlat_dx, dlat_dy), np.arctan2(dlon_dx, -dlon_dy)


def compute_drifters_thetas_core(lonc, latc, phi, lonxy, latxy, x, y, dx=100, dy=100):
    """Compute the local angles theta between e_lon and x and between e_lat and x for drifters locations of one colocalisation
    https://github.com/rasterio/affine

    Parameters
    ----------
    lonc, latc, phi: float
    central position and orientation of the box
    lonxy, latxy : drifters position in longitude, latitude
    x, y: np.array position of drifters in local coordinates (with origin at (lonc, latc) and x-axis aligned
    with the satellite track (lonc, latc) - (lon1, lat1) direction)
    dx, dy : differential, default is 100m

    Return
    ------
    theta_longitude, theta_latitude : np.array, np.array


    """
    proj = get_proj(lonc, latc)
    # assert False, help(proj.transform)
    xc, yc = proj.transform(lonc, latc)
    # apply inverse affine transformation
    a_fwrd = Affine.translation(-xc, -yc) * Affine.rotation(-phi, pivot=(xc, yc))
    a_back = ~a_fwrd
    # compute coordinates of x,y in the lon, lat orientated grid
    x_invx, y_invx = a_back * (x + dx, y)
    lonx, latx = proj.transform(
        x_invx,
        y_invx,
        direction=pyproj.enums.TransformDirection.INVERSE,
    )
    dlon_dx, dlat_dx = (lonx - lonxy) / dx, (latx - latxy) / dx

    x_invy, y_invy = a_back * (x, y + dy)
    lony, laty = proj.transform(
        x_invy,
        y_invy,
        direction=pyproj.enums.TransformDirection.INVERSE,
    )
    dlon_dy, dlat_dy = (lony - lonxy) / dy, (laty - latxy) / dy

    theta_lon = np.arctan2(-dlat_dx, dlat_dy)  # angles between e_lon and x
    theta_lat = np.arctan2(dlon_dx, -dlon_dy)  # angles between e_lat and x
    return theta_lon, theta_lat


def compute_drifters_thetas(ds, dx=100, dy=100):
    """Compute the local angles theta between e_lon and x and between e_lat and x for drifters locations for all colocalisations

     Parameters
    ----------
    ds : xr.DataArray
        colocalisations dataset
    dx, dy : differential, default is 100m

    Return
    ------
    theta_lon, theta_latitude : xr.Datarray, xr.Datarray

    """
    tdim = "site_obs"
    theta_lon, theta_lat = xr.apply_ufunc(
        compute_drifters_thetas_core,
        ds.box_lonc,
        ds.box_latc,
        ds.box_phi,
        ds.drifter_lon,
        ds.drifter_lat,
        ds.drifter_x,
        ds.drifter_y,
        dx,
        dy,
        input_core_dims=[[]] * 3 + [[tdim]] * 4 + [[]] * 2,
        output_core_dims=[[tdim]] * 2,
        vectorize=True,
        dask="parallelized",
    )
    return theta_lon, theta_lat


""" 
VE, VN -> VX,VY
--------------------------------------------
"""


def vevn2vxvy(theta_lon, theta_lat, ve, vn):
    """Compute velocities projected on the local box grid (ve, vn -> vx-along satellite track, vy-normal to satellite track)
    https://github.com/rasterio/affine

    Parameters
    ----------
    lonc, latc, phi: float  central position and orientation of the box
    ve,vn: np.array velocities on lon, lat grid
    theta_lon, theta lat

    Return
    ------
    local vx, local vy : np.array, np.array

    """
    return ve * np.cos(theta_lon) + vn * np.cos(theta_lat), ve * np.sin(
        theta_lon
    ) + vn * np.sin(theta_lat)


def compute_local_drifters_velocities(ds, tdim):
    """Compute local coordinates lon, lat of the local grid of the box (x-along satellite track, y-normal to satellite track -> lon,lat) for all colocalisations

     Parameters
    ----------
    ds : xr.DataArray
        colocalisations dataset
    suff : str
        data prefix
    tdim : str
        time dimension of data with prefix suff

    Return
    ------
    local vx, local vy : xr.Datarray, xr.Datarray

    """
    ve = ds["drifter_ve"]
    vn = ds["drifter_vn"]

    # will need to vectorize with dask eventually
    vx, vy = xr.apply_ufunc(
        vevn2vxvy,
        ds.drifter_theta_lon,
        ds.drifter_theta_lat,
        ve,
        vn,
        input_core_dims=[[tdim]] * 2 + [[tdim]] * 2,
        output_core_dims=[[tdim]] * 2,
        vectorize=True,
        dask="parallelized",
    )
    return vx, vy
    
    
"""
BUILDING DATASET AND ADD THE BOX velocities, sla gradients
---------------------------------------------------------------------------------------------------------
"""
# def build_dataset(nc,
def build_dataset(
    nc,
    x=np.arange(-300e3, 300e3, 5e3),
    y=np.arange(-200e3, 200e3, 5e3),
    chunks=None,
    persist=False,
):
    """Build a dataset from data from one .nc file and add the box

     Parameters
    ----------
    nc : str
        path to netcdf file to open
    x,y : np.array, np.array
        local coordinates of the box, default is x = np.arange(-300e3,300e3,5e3) and y = np.arange(-200e3,200e3,5e3)
    persist : bool
            True to return ds.persist

    Return
    ------
    ds : xr.Datarray
        dataset with data of one .nc file and the local box added

    """
    if chunks is None:
        chunks = dict(obs=1000)
    # ds = (xr.open_dataset(nc)
    ds = xr.open_dataset(nc).chunk(chunks)

    # prevent to use apply_ufuncs(vectorize=True) on 1 chunk dataset
    if ds.dims['obs']==1 :
        print('obs dim =1')
        return 1
    elif ds.dims['obs']<=1000:
        ds = ds.chunk({'obs':ds.dims['obs']//2})
        
        
    # change prefixes
    ds = change_prefix(ds)

    #put longitude in the -180-180° system for all
    from histlib.cstes import lon_360_to_180
    if (ds["lon"] > 180).any() or (ds["lon"] < -180).any():
        ds['lon'] = lon_360_to_180(ds.lon)
        print('lon from 0-360° to -180-180°')
    if (ds["alti_lon"] > 180).any() or (ds["alti_lon"] < -180).any():
        ds['alti_lon'] = lon_360_to_180(ds.alti_lon)
        print('alti_lon from 0-360° to -180-180°')
    if (ds["drifter_lon"] > 180).any() or (ds["drifter_lon"] < -180).any():
        ds['drifter_lon'] = lon_360_to_180(ds.drifter_lon)
        print('drifter_lon from 0-360° to -180-180°')
    
    # add several variables in coords
    ds = ds.set_coords(["drifter_" + d for d in ["time", "lon", "lat"]]).set_coords(
        ["alti_" + d for d in ["time_", "lon", "lat"]]
    )
    #put longitude in the -180-180° system
    
    # ds = ds.persist()

    # add box
    ds["box_lonc"], ds["box_latc"], ds["box_phi"] = compute_box(ds)
    ds["box_lon"], ds["box_lat"] = compute_box_grid_lonlat(ds, x, y)
    ds["drifter_x"], ds["drifter_y"] = compute_local_xy(ds, "drifter", "site_obs")
    ds = ds.set_coords(["drifter_x", "drifter_y"])
    ds["alti_x"], ds["alti_y"] = compute_local_xy(ds, "alti", "alti_time")
    ds = ds.set_coords(["alti_x", "alti_y"])

    # thetas
    ds["box_theta_lon"], ds["box_theta_lat"] = compute_box_thetas(ds)
    ds["drifter_theta_lon"], ds["drifter_theta_lat"] = compute_drifters_thetas(ds)

    # project velocities on the local box axes
    ds["drifter_vx"], ds["drifter_vy"] = compute_local_drifters_velocities(
        ds, "site_obs"
    )

    # add along track sla gradients

    t = ds["alti_time"]
    tc = (t + t.shift({"alti_time": -1})) * 0.5  # C
    tc = tc[:-1].rename({"alti_time": "alti_time_mid"})

    ds = ds.assign_coords(**{"alti_time": t, "alti_time_mid": tc})
    grid = Grid(ds, coords={"t": {"center": "alti_time_mid", "outer": "alti_time"}})

    ds["alti_x_mid"] = grid.interp(ds["alti_x"], axis="t")
    ds["alti_y_mid"] = grid.interp(ds["alti_y"], axis="t")
    ds = ds.set_coords(["alti_x_mid", "alti_y_mid"])

    # dt = grid.diff(ds[asuff+"_time_"], axis="t")/pd.Timedelta("1S")
    dx = grid.diff(ds["alti_x"], axis="t")
    dy = grid.diff(ds["alti_y"], axis="t")
    dxy = np.sqrt(dx**2 + dy**2)
    
    g = 9.81
    listv = [l for l in list(ds.variables) if 'sla' in l]+['alti_mdt','alti_ocean_tide', 'alti_dac', 'alti_internal_tide']
    for sla in listv :
        dsla = grid.diff(ds[sla], axis="t")
        ds[sla.replace('alti', 'alti_ggx')] = g * dsla / dxy

    # add Coriolis frequency
    ds["f"] = 2 * 2 * np.pi / 86164.1 * np.sin(ds.lat * np.pi / 180)

    # add momentum conservation equation term
    dt = 3600
    ds["drifter_acc_x"] = ds.drifter_vx.differentiate("site_obs")/dt
    ds["drifter_acc_y"] = ds.drifter_vy.differentiate("site_obs")/dt
    ds["drifter_coriolis_x"] = -ds["drifter_vy"] * ds.f
    ds["drifter_coriolis_y"] = ds["drifter_vx"] * ds.f

    #add coord for site_obs to simplify concatenation in case site_obs dim have different lengths
    ds = ds.assign_coords(site_obs=np.arange(ds.dims['site_obs']))

    # attrs
    ds.drifter_x.attrs = {
        "description": "drifter's along track direction x position on the box",
        "units": "m",
    }
    ds.drifter_y.attrs = {
        "description": "drifter's cross track direction y position on the box",
        "units": "m",
    }
    ds.drifter_lon.attrs = {"description": "drifter's longitude", "units": "°"}
    ds.drifter_lat.attrs = {"description": "drifter's latitude", "units": "°"}

    ds.alti_x.attrs = {
        "description": "altimeter's along track direction x position on the box",
        "units": "m",
    }
    ds.alti_y.attrs = {
        "description": "altimeter's cross track direction y position on the box",
        "units": "m",
    }

    ds.box_x.attrs = {
        "description": "along track direction x coordinate of the box",
        "units": "m",
    }
    ds.box_y.attrs = {
        "description": "cross track direction y coordinate of the box",
        "units": "m",
    }

    ds.drifter_vx.attrs = {
        "description": "drifter's along track direction velocity on the box",
        "units": r"$m.s^{-1}$",
        "long_name": "u",
    }
    ds.drifter_vy.attrs = {
        "description": "drifter's cross track direction velocity on the box",
        "units": r"$m.s^{-1}$",
        "long_name": "v",
    }
    ds.drifter_ve.attrs = {
        "description": "drifter's eastward velocity",
        "units": r"$m.s^{-1}$",
        "long_name": r"$u_{eastward}$",
    }
    ds.drifter_vn.attrs = {
        "description": "drifter's northward velocity",
        "units": r"$m.s^{-1}$",
        "long_name": r"$v_{northward}$",
    }

    ds.drifter_acc_x.attrs = {
        "description": "drifter's along track direction acceleration on the box",
        "units": r"$m.s^{-2}$",
        "long_name": r"$d_tu$",
    }
    ds.drifter_acc_y.attrs = {
        "description": "drifter's cross track direction acceleration on the box",
        "units": r"$m.s^{-2}$",
        "long_name": r"$d_tv$",
    }

    ds.drifter_coriolis_x.attrs = {
        "description": "along track coriolis term from drifter",
        "units": r"$m.s^{-2}$",
        "long_name": r"$-fv$",
    }
    ds.drifter_coriolis_y.attrs = {
        "description": "cross track coriolis term from drifter",
        "units": r"$m.s^{-2}$",
        "long_name": r"$fu$",
    }

    ds.drifter_time.attrs = {"description": "drifter's trajectory time measurements"}

    #ds.alti_g_grad_x.attrs = {
    #    "description": "cross track sla gradient term from the altimeter",
    #    "units": r"$m.s^{-2}$",
    #    "long_name": r"$g\partial_x\eta$",
    #}
    #ds.alti_denoised_g_grad_x.attrs = {
    #    "description": "denoised cross track sla gradient term from the altimeter",
    #    "units": r"$m.s^{-2}$",
    #    "long_name": r"$g\partial_x\eta$",
    #}

    ds.box_lat.attrs = {
        "description": "latitudes of box points",
        "units": "°",
    }
    ds.box_lon.attrs = {
        "description": "longitudes of box points",
        "units": "°",
    }
    ds.box_latc.attrs = {
        "description": "latitude of the center of the track, the altimeter's matchup point",
        "units": "°",
    }
    ds.box_lonc.attrs = {
        "description": "longitude of the center of the track, the altimeter's matchup point",
        "units": "°",
    }
    ds.box_phi.attrs = {
        "description": "angle between the track and the longitude axis",
        "units": "°",
    }
    ds.box_theta_lat.attrs = {
        "description": "local angle between the track and the longitude axis for box's points",
        "units": "°",
    }
    ds.box_theta_lon.attrs = {
        "description": "local angle between the track and the latitude axis for box's points",
        "units": "°",
    }

    ds.drifter_theta_lat.attrs = {
        "description": "local angle between the track and the longitude axis along the drifter's trajectory",
        "units": "°",
    }
    ds.drifter_theta_lon.attrs = {
        "description": "local angle between the track and the latitude axis along the drifter's trajectory",
        "units": "°",
    }
    ds.f.attrs = {
        "description": "coriolis frequency at the altimeter's matchup point",
        "units": r"$s^{-1}$",
    }

    if persist:
        ds = ds.persist()

    return ds


"""
ERRORS
---------------------------------------------------------------------------------------------------------
"""
def dlonlat_dlonlatkm(dlon, dlat, lat):
    """ return the error distance in m from the error in degree"""
    R=6371e3
    return R*np.arccos(np.sin(lat)**2 + (np.cos(lat)**2)*np.cos(dlon)), R*dlat
                       
def rotation(dlonkm, dlatkm, box_phi):#APPROXIMATIF
    """ apply rotation lon, lat -> xy """ 
    return dlonkm*np.cos(box_phi)+dlatkm*np.sin(box_phi), -dlonkm*np.sin(box_phi)+dlatkm*np.cos(box_phi) 
def err_xy_obs(dobs):
    #_drop = ['alti_lat','alti_lon','alti_time','alti_time_','alti_time_mid','alti_x','alti_x_mid','alti_y','alti_y_mid','drifter_lat','drifter_lon','drifter_time','drifter_x','drifter_y',]
    #dobs= dobs.drop(_drop)
    dlon=dobs.drifter_err_lon.compute()*np.pi/180 
    dlat=dobs.drifter_err_lat.compute()*np.pi/180
    lat=dobs.lat.compute()*np.pi/180
    box_phi = dobs.box_phi.compute()*np.pi/180
    dlonkm, dlatkm=dlonlat_dlonlatkm(dlon, dlat,lat) #need computed values !!!
    
    ds=xr.Dataset()
    ds['drifter_err_lonm']=dlonkm.assign_attrs({"long_name":r'$err_{lon}$', "units":'m'})
    ds['drifter_err_latm']=dlatkm.assign_attrs({"long_name":r'$err_{lat}$', "units":'m'})
    errx, erry = rotation(dlonkm, dlatkm, box_phi)
    ds['drifter_err_x']=errx.assign_attrs({"long_name":r'$err_x$', "units":'m'})
    ds['drifter_err_y']=erry.assign_attrs({"long_name":r'$err_y$', "units":'m'})
    
    errvx, errvy = vevn2vxvy(dobs.drifter_theta_lon, dobs.drifter_theta_lat, dobs.drifter_ve, dobs.drifter_vn)
    ds['drifter_err_vx']=errvx.assign_attrs({"long_name":r'$err_{vx}$', "units":'m/s'})
    ds['drifter_err_vy']=errvy.assign_attrs({"long_name":r'$err_{vy}$', "units":'m/s'})
    return ds
    
def _concat_err_xy(ds):
    """ concatenate """
    return xr.concat([err_xy_obs(ds.sel(obs=o)) for o in ds.obs], 'obs')

def compute_err_xy(ds):
    """
    https://xarray.pydata.org/en/stable/user-guide/dask.html
    Parameters
    ----------
    ds: xr.Dataset
        Input collocation dataset
    
    Return
    ------
    ds: xr.Dataset
        Dataset with errors

    """
    #_drop = ['alti_lat','alti_lon','alti_time','alti_time_','alti_time_mid','alti_x','alti_x_mid','alti_y','alti_y_mid','drifter_lat','drifter_lon','drifter_time','drifter_x','drifter_y',]
    #ds=ds.drop(_drop)

    # build template
    template = _concat_err_xy(ds.isel(obs=slice(0,2)))
    
    # broad cast along obs dimension
    template, _ = xr.broadcast(template.isel(obs=0).chunk(), 
                               ds.time,
                               exclude=[ "box_x", "box_y",],
                              )
    
    # massage further and unify chunks
    template = (xr.merge([ds[['lon','lat', 'time']], template.drop(list(template.coords))])
             .unify_chunks()[list(template.keys())])
    
    # actually perform the calculation
    try : 
        ds_err = xr.map_blocks(_concat_err_xy, ds, template=template).compute()
    except : 
        assert False, 'pb map_blocks'
   
    return ds_err