import os
from glob import glob


import xarray as xr
import pandas as pd
import numpy as np
import histlib.cstes as cstes

# directories

glophy_dir = "/home/ref-cmems-public/tac/multiobs/MULTIOBS_GLO_PHY_REP_015-004/dataset-uv-rep-hourly/"

"""
ERA STAR  
---------------------------------------------------------------------------------------------------------
"""


def load_eras(t, dt=None, suffix="es_", to_360=False, rkwargs=None, **kwargs):
    """Extract ERA star data
    Ref ...

    Parameters
    ----------
    t: date-like object
        date for extraction
    dt: tuple, optional
        time interval in hours around t
    """
    # TIME
    t = pd.to_datetime(t)
    if dt is not None:#select several hours around matchup
        t = [t + pd.Timedelta(np.timedelta64(_dt, "h")) for _dt in range(*dt)]
    else:
        t = [t]
    if rkwargs is None:
        rkwargs = {}
    # LOAD
    D = []
    for _t in t:
        _rpath = f"{_t.year}/{_t.dayofyear:03d}/{_t.year}{_t.month:02d}{_t.day:02d}{_t.hour:02d}*.nc"
        files = glob(os.path.join(era_star_dir, _rpath))
        print(files, _rpath)

        if to_360:  # turn erastar to 0-360° if needed
            _ds = (
                xr.load_dataset(files[0], **rkwargs)
                # .sel(**kwargs) #in case 180 limit
            )
            assert all(
                _ds.lon <= 180
            ), "error : era_star in 0-360° lon coordinates"  # verify -180-180° representation for lon coordinates
            _ds = _ds.assign_coords(lon=cstes.lon_180_to_360(_ds.lon))
            _ds = _ds.sortby("lon")
            _ds = _ds.sel(**kwargs)  # select data around the colocation

        else:
            _ds = xr.load_dataset(files[0], **rkwargs).sel(**kwargs)

            assert all(
                _ds.lon <= 180
            ), "error : era_star in 0-360° lon coordinates"  # verify -180-180° representation for lon coordinates

        D.append(_ds)
    ds = xr.concat(D, dim="time")
    ds = ds.rename(lon=suffix + "lon", lat=suffix + "lat", time=suffix + "time")

    return ds


"""
Build dataset  
-------------------------------------------
"""


def get_eras_one_obs(ds_obs, dt=(-12, 13), only_matchup_time=True):
    """load erastar for one colocalization"""
    dl = 0.125
    assert (ds_obs["box_lon"] <= 180).all(), "error : ds_obs in 0-360° lon coordinates"

    # -180-180 longitude problem
    limite_lon = (
        (ds_obs.drifter_lon > 179).any()
        or (ds_obs.drifter_lon < -179).any()
        or (ds_obs.box_lon > 179).any()
        or (ds_obs.box_lon < -179).any()
    )

    # box_dim = float( 1.5*ds_obs.box_x.max()*1e-3/111/np.cos(2*np.pi*ds_obs.box_latc/360) ) # *
    # limite_lon = ds_obs.box_lonc.values > 180-box_dim or ds_obs.box_lonc.values < -(180-box_dim)

    if limite_lon:
        box_lon = cstes.lon_180_to_360(ds_obs.box_lon)
        drifter_lon = cstes.lon_180_to_360(ds_obs.drifter_lon)
        _alti_lon = cstes.lon_180_to_360(
            ds_obs.alti_lon.isel(alti_time=ds_obs.dims['alti_time']//2)
        )  # only matchup

    else:
        box_lon = ds_obs["box_lon"]
        drifter_lon = ds_obs["drifter_lon"]
        _alti_lon = ds_obs["alti_lon"].isel(alti_time=ds_obs.dims['alti_time']//2)  # only matchup

    # matchup site
    site_matchup_indice = int(ds_obs.__site_matchup_indice.values)
    _drifter_lon = drifter_lon.isel(site_obs=site_matchup_indice).values
    _drifter_lat = ds_obs.drifter_lat.isel(site_obs=site_matchup_indice).values
    _drifter_time = ds_obs.drifter_time.isel(site_obs=site_matchup_indice).values

    # select eras only over the box/drifter trajectory
    lon_min = min(box_lon.min(), drifter_lon.min())
    lon_max = max(box_lon.max(), drifter_lon.max())
    lat_min = min(ds_obs.box_lat.min(), ds_obs.drifter_lat.min())
    lat_max = max(ds_obs.box_lat.max(), ds_obs.drifter_lat.max())

    # load data
    _drop = ["es_u10s", "es_v10s", "e5_u10s", "e5_v10s", "count", "quality_flag"]
    try:
        ds_eras = load_eras(
            ds_obs.time.values,
            dt,
            to_360=limite_lon,
            rkwargs={"drop_variables": _drop},
            lon=np.arange(lon_min - dl, lon_max + dl, dl),
            lat=np.arange(lat_min - dl, lat_max + dl, dl),
            method="nearest",
            tolerance=dl,
        )
        ds_eras = ds_eras.rename(
            {v: v.replace("tauu", "taue") for v in ds_eras if "tauu" in v}
        ).rename(
            {v: v.replace("tauv", "taun") for v in ds_eras if "tauv" in v}
        )  # NEW
    except:
        assert False, ds_obs.__site_id

    # interplate over the trajectoire
    ds_traj = (
        ds_eras.interp(
            es_time=ds_obs.drifter_time, es_lon=drifter_lon, es_lat=ds_obs.drifter_lat
        )
        .rename({v: "traj_" + v for v in ds_eras})
        .drop(
            [
                "drifter_lon",
                "drifter_lat",
                "es_time",
                "es_lon",
                "es_lat",
                "lat",
                "lon",
                "time",
            ]
        )
    )

    # at drifter matchup
    ds_drifter_matchup = (
        ds_traj.isel(site_obs=site_matchup_indice)
        .drop(list(ds_traj.coords.keys()))
        .rename({v: v.replace("traj_", "drifter_matchup_") for v in ds_traj})
    )  # NEW

    # interpolate on the box
    if only_matchup_time:
        ds_box = (
            ds_eras.interp(es_time=_drifter_time, es_lon=box_lon, es_lat=ds_obs.box_lat)
            .rename({v: "box_" + v for v in ds_eras})
            .drop(["es_lon", "es_lat", "es_time", "lat", "lon", "time"])
        )
    else:
        ds_box = (
            ds_eras.interp(es_lon=box_lon, es_lat=ds_obs.box_lat)
            .rename({v: "box_" + v for v in ds_eras})
            .drop(["es_lon", "es_lat", "lat", "lon", "time"])
        )

    # interpolate at drifter matchup position, for all drifter times
    # ds_drifter = ds_eras.interp(es_lon = _drifter_lon, es_lat = _drifter_lat).drop(['es_lon', 'es_lat']).rename({v: "drifter_matchup_"+v for v in ds_eras})
    ds_drifter = (
        ds_eras.interp(es_lon=_drifter_lon, es_lat=_drifter_lat)
        .drop(["es_lon", "es_lat"])
        .rename({v: "drifter_temp_" + v for v in ds_eras})
    )

    # at alti matchup
    ds_alti = ds_eras.interp(
        es_time=ds_obs.alti_time_.isel(alti_time=ds_obs.dims['alti_time']//2),
        es_lon=_alti_lon,
        es_lat=ds_obs.alti_lat.isel(alti_time=ds_obs.dims['alti_time']//2),
    )
    ds_alti = ds_alti.drop(list(ds_alti.coords.keys())).rename(
        {v: "alti_matchup_" + v for v in ds_eras}
    )

    # convert to dataset and massage in order to concatenate properly in get_aviso
    ds = (
        (xr.merge([ds_traj, ds_drifter_matchup, ds_box, ds_drifter, ds_alti]))
        .reset_index("es_time")
        .reset_coords(["drifter_time", "drifter_x", "drifter_y", "es_time_"])
    )  # NEW

    #ds["time"] = ds_obs.time.drop(["lon", "lat"])
    ds['obs'] = ds_obs.obs
    ds = ds.drop(['time', 'lon', 'lat'])
    ds = ds.rename({v: "es_" + v.replace("_es", "") for v in ds if "_es" in v})
    ds = ds.rename({v: "e5_" + v.replace("_e5", "") for v in ds if "_e5" in v})

    return ds


def _concat_eras(ds, dt, only_matchup_time=True):
    """Load erastar data for multiple colocalizations and concatenate"""
    try:
        D = []
        for o in ds.obs:
            D.append(get_eras_one_obs(ds.sel(obs=o), dt, only_matchup_time))
        _ds = xr.concat(D,"obs")
    except:
        assert False, (ds.__site_id.values, ds.obs.values)
    return _ds


def compute_eras(ds, dt, only_matchup_time=True):
    """
    https://xarray.pydata.org/en/stable/user-guide/dask.html
    Parameters
    ----------
    ds: xr.Dataset
        Input colocation dataset
    dt: tuple
        Time offsets compared to the colocation, e.g. if colocation
        is at time t and dt=(-12,13), erastar data will be interpolated on
        the interval (t-12, t+12 hours) with an hour step
        Default is (-12,13)
    only_matchup_time : bool
                        if True return stress on the box only for the drifter matchup time
                        if False return it for all erastar time over the dt period

    Return
    ------
    ds: xr.Dataset
        Dataset containing relevant aviso data

    """
    # build template
    template = _concat_eras(
        ds.isel(obs=slice(0, 2)), dt=dt, only_matchup_time=only_matchup_time
    )

    # broad cast along obs dimension
    template, _ = xr.broadcast(
        template.isel(obs=0).chunk(),
        ds,
        exclude=[
            "es_time",
            "alti_time",
            "alti_time_mid",
            "site_obs",
            "box_x",
            "box_y",
        ],
    )

    # massage further and unify chunks
    template = xr.merge(
        [
            template,
            ds.drop(
                [
                    "lon",
                    "lat",
                    "time",
                    "drifter_lon",
                    "drifter_lat",
                    "drifter_x",
                    "drifter_y",
                    "drifter_time",
                ]
            ),
        ]
    ).unify_chunks()[list(template.keys())]

    # dimension order is not consistent with that of exiting from map_blocks for some reason
    dims = ["obs", "es_time", "box_y", "box_x", "site_obs"]

    template = template.transpose(
        *dims
    )  # .set_coords(["drifter_time", "drifter_x","drifter_y"])

    # actually perform the calculation
    ds_eras = xr.map_blocks(
        _concat_eras,
        ds,
        kwargs=dict(dt=dt, only_matchup_time=only_matchup_time),
        template=template,
    )
    ds_eras = ds_eras.set_coords(["obs", "es_time_"])  # coordinate for obs dimension

    """ NOT HERE (COMPUTE SITE OBS MATCHUP INDICE AGAIN)
    #add drifter matchup
    __site_matchup_indice = ds.__site_matchup_indice.compute()
    e5_drifter_matchup_taue = ds_eras.e5_traj_taue.isel(site_obs = __site_matchup_indice)
    e5_drifter_matchup_taun = ds_eras.e5_traj_taun.isel(site_obs = __site_matchup_indice)
    es_drifter_matchup_taue = ds_eras.es_traj_taue.isel(site_obs = __site_matchup_indice)
    es_drifter_matchup_taun = ds_eras.es_traj_taun.isel(site_obs = __site_matchup_indice)
    ds_eras['e5_drifter_matchup_taue'] = ds_eras.e5_traj_taue.isel(site_obs = __site_matchup_indice)
    ds_eras['e5_drifter_matchup_taun'] = ds_eras.e5_traj_taun.isel(site_obs = __site_matchup_indice)
    ds_eras['es_drifter_matchup_taue'] = ds_eras.es_traj_taue.isel(site_obs = __site_matchup_indice)
    ds_eras['es_drifter_matchup_taun'] = ds_eras.es_traj_taun.isel(site_obs = __site_matchup_indice)
    """
    # attrs
    ds_eras.e5_alti_matchup_taue.attrs = {
        "long_name": "era5 eastward wind stress interpolated on the altimeter's matchup",
        "units": "Pa",
    }
    ds_eras.e5_alti_matchup_taun.attrs = {
        "long_name": "era5 northward wind stress interpolated on the altimeter's matchup",
        "units": "Pa",
    }

    ds_eras.e5_drifter_matchup_taue.attrs = {
        "long_name": "era5 eastward wind stress interpolated on the drifter's matchup",
        "units": "Pa",
    }
    ds_eras.e5_drifter_matchup_taun.attrs = {
        "long_name": "era5 northward wind stress interpolated on the drifter's matchup",
        "units": "Pa",
    }

    ds_eras.e5_box_taue.attrs = {
        "long_name": "era5 eastward wind stress interpolated on the box at the matchup time or several times (depending on the dimension)",
        "units": "Pa",
    }
    ds_eras.e5_box_taun.attrs = {
        "long_name": "era5 northward wind stress interpolated on the box at the matchup time or several times (depending on the dimension)",
        "units": "Pa",
    }

    ds_eras.e5_drifter_temp_taue.attrs = {
        "long_name": "era5 eastward wind stress interpolated on the drifter's matchup position for several times (static but temporal variation)",
        "units": "Pa",
    }
    ds_eras.e5_drifter_temp_taun.attrs = {
        "long_name": "era5 northward wind stress interpolated on the drifter's matchup position for several times (static but temporal variation)",
        "units": "Pa",
    }

    ds_eras.e5_traj_taue.attrs = {
        "long_name": "era5 eastward wind stress interpolated on the drifter's trajectory",
        "units": "Pa",
    }
    ds_eras.e5_traj_taun.attrs = {
        "long_name": "era5 northward wind stress interpolated on the drifter's trajectory",
        "units": "Pa",
    }

    ds_eras.es_alti_matchup_taue.attrs = {
        "long_name": "erastar eastward wind stress interpolated on the altimeter's matchup",
        "units": "Pa",
    }
    ds_eras.es_alti_matchup_taun.attrs = {
        "long_name": "erastar northward wind stress interpolated on the altimeter's matchup",
        "units": "Pa",
    }

    ds_eras.es_drifter_matchup_taue.attrs = {
        "long_name": "erastar eastward wind stress interpolated on the drifter's matchup",
        "units": "Pa",
    }
    ds_eras.es_drifter_matchup_taun.attrs = {
        "long_name": "erastar northward wind stress interpolated on the drifter's matchup",
        "units": "Pa",
    }

    ds_eras.es_box_taue.attrs = {
        "long_name": "erastar eastward wind stress interpolated on the box at the matchup time",
        "units": "Pa",
    }
    ds_eras.es_box_taun.attrs = {
        "long_name": "erastar northward wind stress interpolated on the box at the matchup time",
        "units": "Pa",
    }

    ds_eras.es_drifter_temp_taue.attrs = {
        "long_name": "erastar eastward wind stress interpolated on the drifter's matchup position for several times (static but temporal variation)",
        "units": "Pa",
    }
    ds_eras.es_drifter_temp_taun.attrs = {
        "long_name": "erastar northward wind stress interpolated on the drifter's matchup position for several times (static but temporal variation)",
        "units": "Pa",
    }

    ds_eras.es_traj_taue.attrs = {
        "long_name": "erastar eastward wind stress interpolated on the drifter's trajectory",
        "units": "Pa",
    }
    ds_eras.es_traj_taun.attrs = {
        "long_name": "erastar northward wind stress interpolated on the drifter's trajectory",
        "units": "Pa",
    }

    return ds_eras
