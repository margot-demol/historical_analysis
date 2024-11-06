import os
from glob import glob


import xarray as xr
import pandas as pd
import numpy as np

import histlib.cstes as cstes

# alti
aviso_dir = "/home/ref-cmems-public/tac/sea-level/SEALEVEL_GLO_PHY_L4_MY_008_047/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D/"

"""
AVISO  
---------------------------------------------------------------------------------------------------------
"""


def load_aviso(t, dt=None, suffix="aviso_", to_360=False, rkwargs=None, **kwargs):
    """Extract AVISO data
    Ref ..

    Parameters
    ----------
    t: date-like object
        date for extraction
    dt: tuple, optional
        time interval in days around t
    rkwargs: dict
        dictionary passed to xr.open_dataset
    **kwargs: dict
        passed to sel
    """

    t = pd.to_datetime(t)
    if dt is not None:
        t = [t + pd.Timedelta(days=_dt) for _dt in range(*dt)]
    else:
        t = [t]
    if rkwargs is None:
        rkwargs = {}
    # LOADftry
    D = []
    for _t in t:
        try:  # ds_obsselected with sel, ds_obs is numpy.datetime64
            files = glob(os.path.join(aviso_dir, f"{_t.year}/{_t.dayofyear:03d}/*.nc"))
        except:  # robust if ds_obs selected with where (ds_obs.time.values  = DatetimeIndex)
            files = glob(
                os.path.join(
                    aviso_dir, f"{_t.year.values[0]}/{_t.dayofyear.values[0]:03d}/*.nc"
                )
            )
            print("need values[0] for load aviso")
        assert len(files) == 1, f"error: multiple files at {files}"

        if to_360:  # turn aviso to 0-360° if needed
            _ds = (
                xr.load_dataset(files[0], **rkwargs)
                # .sel(**kwargs) #in case 180 limit
            )

            assert all(
                _ds.longitude <= 180
            ), "error : aviso in 0-360° lon coordinates"  # verify -180-180° representation for lon coordinates
            _ds = _ds.assign_coords(longitude=cstes.lon_180_to_360(_ds.longitude))
            _ds = _ds.sortby("longitude")
            _ds = _ds.sel(**kwargs)  # select data around the colocation

        else:
            _ds = xr.load_dataset(files[0], **rkwargs).sel(**kwargs)

            assert all(
                _ds.longitude <= 180
            ), "error : aviso in 0-360° lon coordinates"  # verify -180-180° representation for lon coordinates

        # ds = xr.open_dataset(files[0], **rkwargs).sel(**kwargs)
        # load_dataset loads data and close the file immediately while read_dataset is lazy
        # https://docs.xarray.dev/en/latest/generated/xarray.load_dataset.html
        # this choice may be more safe with regards to concurrent readings

        D.append(_ds)
    ds = xr.concat(
        D, dim="time", combine_attrs="drop_conflicts"
    )  # drop attributes that vary
    ds = ds.rename(
        time=suffix + "time", longitude=suffix + "lon", latitude=suffix + "lat"
    )
    return ds


"""
Build dataset  
-------------------------------------------
"""


def get_aviso_one_obs(ds_obs, dt=(-1, 2), only_matchup_time=True):
    """load aviso for one colocation"""
    dl = 0.25
    assert (ds_obs["box_lon"] <= 180).all(), "error : ds_obs in 0-360° lon coordinates"

    # -180-180 longitude boundary problem
    limite_lon = (
        (ds_obs.drifter_lon > 178).any()
        or (ds_obs.drifter_lon < -178).any()
        or (ds_obs.box_lon > 178).any()
        or (ds_obs.box_lon < -178).any()
    )

    if limite_lon:
        box_lon = cstes.lon_180_to_360(ds_obs.box_lon)
        drifter_lon = cstes.lon_180_to_360(ds_obs.drifter_lon)
        _alti_lon = cstes.lon_180_to_360(
            ds_obs.alti_lon.isel(alti_time=ds_obs.dims["alti_time"] // 2)
        )  # only matchup

    else:
        box_lon = ds_obs["box_lon"]
        drifter_lon = ds_obs["drifter_lon"]
        _alti_lon = ds_obs["alti_lon"].isel(
            alti_time=ds_obs.dims["alti_time"] // 2
        )  # only matchup

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

    # LOAD DATA
    _drop = [
        "crs",
        "lat_bnds",
        "lon_bnds",
        "ugosa",
        "err_ugosa",
        "vgosa",
        "err_vgosa",
        "ugos",
        "vgos",
        "flag_ice",
        "tpa_correction",
        "nv",
    ]
    try:
        ds_aviso = load_aviso(
            ds_obs.time.values,
            dt,
            suffix="aviso_",
            to_360=limite_lon,
            rkwargs={"drop_variables": _drop},
            longitude=np.arange(lon_min - dl, lon_max + dl, dl),
            latitude=np.arange(lat_min - dl, lat_max + dl, dl),
            method="nearest",
            tolerance=dl,
        )  # aviso sla, adt around matchup_time
    except:
        assert False, (ds_obs.obs.values, ds_obs.__site_id.values, ds_obs.time.values)
    # INTERPOLATIONS
    # for all interpolations, need drop_duplicated, because in some case, same aviso point for two differents longitude, latitude points and interpolation works only one uniquely valued coordinated dataset
    # interpolate on the drifter trajectory

    try:
        ds_traj = (
            ds_aviso.drop_duplicates(["aviso_lon", "aviso_lat"])
            .interp(
                aviso_time=ds_obs.drifter_time,
                aviso_lon=drifter_lon,
                aviso_lat=ds_obs.drifter_lat,
            )
            .rename({v: "aviso_traj_" + v for v in ds_aviso})
            .drop(
                [
                    "drifter_lon",
                    "drifter_lat",
                    "aviso_time",
                    "aviso_lon",
                    "aviso_lat",
                    "lat",
                    "lon",
                    "time",
                ]
            )
        )
    except:
        assert False, (
            "pb interpolate over trajectories",
            ds_obs.obs.values,
            ds_obs.__site_id.values,
            ds_obs.time.values,
        )

    # interpolate on the box
    if only_matchup_time:
        ds_box = (
            ds_aviso.drop_duplicates(["aviso_lon", "aviso_lat"])
            .interp(
                aviso_time=_drifter_time, aviso_lon=box_lon, aviso_lat=ds_obs.box_lat
            )
            .rename({v: "aviso_box_" + v for v in ds_aviso})
            .drop(["aviso_lon", "aviso_lat", "aviso_time", "lat", "lon", "time"])
        )
    else:
        ds_box = (
            ds_aviso.drop_duplicates(["aviso_lon", "aviso_lat"])
            .interp(aviso_lon=box_lon, aviso_lat=ds_obs.box_lat)
            .rename({v: "aviso_box_" + v for v in ds_aviso})
            .drop(["aviso_lon", "aviso_lat", "lat", "lon", "time"])
        )

    # interpolate at drifter matchup position, for all drifter times
    try:
        ds_drifter = (
            ds_aviso.drop_duplicates(["aviso_lon", "aviso_lat"])
            .interp(aviso_lon=_drifter_lon, aviso_lat=_drifter_lat)
            .drop(["aviso_lon", "aviso_lat"])
            .rename({v: "aviso_drifter_temp_" + v for v in ds_aviso})
        )

        # at alti matchup
        ds_alti = ds_aviso.drop_duplicates(["aviso_lon", "aviso_lat"]).interp(
            aviso_time=ds_obs.alti_time_.isel(alti_time=ds_obs.dims["alti_time"] // 2),
            aviso_lon=_alti_lon,
            aviso_lat=ds_obs.alti_lat.isel(alti_time=ds_obs.dims["alti_time"] // 2),
        )
        ds_alti = ds_alti.drop(list(ds_alti.coords.keys())).rename(
            {v: "aviso_alti_matchup_" + v for v in ds_aviso}
        )
    except:
        assert False, (
            "pb interpolate over drifter matchup",
            ds_obs.obs,
            ds_obs.__site_id.values,
            ds_obs.time.values,
        )

    # COMPUTE GRADIENT
    g = 9.81
    try:
        # sla
        ds_box["aviso_box_ggx_sla"] = g * ds_box["aviso_box_sla"].differentiate("box_x")
        ds_box["aviso_box_ggy_sla"] = g * ds_box["aviso_box_sla"].differentiate("box_y")
        ds_traj["aviso_traj_ggx_sla"] = ds_box["aviso_box_ggx_sla"].interp(
            box_x=ds_traj.drifter_x, box_y=ds_traj.drifter_y
        )
        ds_traj["aviso_traj_ggy_sla"] = ds_box["aviso_box_ggy_sla"].interp(
            box_x=ds_traj.drifter_x, box_y=ds_traj.drifter_y
        )
        # adt
        ds_box["aviso_box_ggx_adr"] = g * ds_box["aviso_box_adt"].differentiate("box_x")
        ds_box["aviso_box_ggy_adt"] = g * ds_box["aviso_box_adt"].differentiate("box_y")
        ds_traj["aviso_traj_ggx_adt"] = ds_box["aviso_box_ggx_adt"].interp(
            box_x=ds_traj.drifter_x, box_y=ds_traj.drifter_y
        )
        ds_traj["aviso_traj_ggy_adt"] = ds_box["aviso_box_ggy_adt"].interp(
            box_x=ds_traj.drifter_x, box_y=ds_traj.drifter_y
        )
        # ds_traj['aviso_traj_ggx']= g * ds_traj['aviso_traj_sla'].differentiate('drifter_x')
        # ds_traj['aviso_traj_ggy']= g * ds_traj['aviso_traj_sla'].differentiate('drifter_y')

    except:
        try:
            # ds_traj['aviso_traj_ggx']= g * ds_traj['aviso_traj_sla'].compute().differentiate('drifter_x')
            # ds_traj['aviso_traj_ggy']= g * ds_traj['aviso_traj_sla'].compute().differentiate('drifter_y')
            ds_box["aviso_box_ggx_sla"] = g * ds_box[
                "aviso_box_sla"
            ].compute().differentiate("box_x")
            ds_box["aviso_box_ggy_sla"] = g * ds_box[
                "aviso_box_sla"
            ].compute().differentiate("box_y")
            ds_traj["aviso_traj_ggx_sla"] = ds_box["aviso_box_ggx_sla"].interp(
                box_x=ds_traj.drifter_x, box_y=ds_traj.drifter_y
            )
            ds_traj["aviso_traj_ggy_sla"] = ds_box["aviso_box_ggy_sla"].interp(
                box_x=ds_traj.drifter_x, box_y=ds_traj.drifter_y
            )
            # adt
            ds_box["aviso_box_ggx_adt"] = g * ds_box[
                "aviso_box_adt"
            ].compute().differentiate("box_x")
            ds_box["aviso_box_ggy_adt"] = g * ds_box[
                "aviso_box_adt"
            ].compute().differentiate("box_y")
            ds_traj["aviso_traj_ggx_adt"] = ds_box["aviso_box_ggx_adt"].interp(
                box_x=ds_traj.drifter_x, box_y=ds_traj.drifter_y
            )
            ds_traj["aviso_traj_ggy_adt"] = ds_box["aviso_box_ggy_adt"].interp(
                box_x=ds_traj.drifter_x, box_y=ds_traj.drifter_y
            )
        except:
            assert False, "pb differentiate grad"
    # sla
    ds_alti
    ds_alti["aviso_alti_matchup_ggx_sla"] = ds_box["aviso_box_ggx_sla"].sel(
        box_x=0, box_y=0
    )
    ds_alti["aviso_alti_matchup_ggy_sla"] = ds_box["aviso_box_ggy_sla"].sel(
        box_x=0, box_y=0
    )
    # adt
    ds_alti["aviso_alti_matchup_ggx_adt"] = ds_box["aviso_box_ggx_adt"].sel(
        box_x=0, box_y=0
    )
    ds_alti["aviso_alti_matchup_ggy_adt"] = ds_box["aviso_box_ggy_adt"].sel(
        box_x=0, box_y=0
    )

    # at drifter matchup
    ds_drifter_matchup = ds_traj[[v for v in ds_traj if "gg" in v]].isel(
        site_obs=site_matchup_indice
    )
    ds_drifter_matchup = ds_drifter_matchup.rename(
        {v: v.replace("traj_", "drifter_matchup_") for v in ds_drifter_matchup}
    ).drop(
        list(ds_drifter_matchup.coords.keys())
    )  # NEW

    # convert to dataset and massage in order to concatenate properly in get_aviso
    ds = (
        (xr.merge([ds_traj, ds_drifter_matchup, ds_box, ds_drifter, ds_alti]))
        .reset_index("aviso_time")
        .reset_coords(["drifter_time", "drifter_x", "drifter_y", "aviso_time_"])
    )

    # ds["time"] = ds_obs.time.drop(["lon", "lat"])
    ds["obs"] = ds_obs.obs.drop(["lon", "lat"])
    ds = ds.drop("time")
    return ds


def _concat_aviso(ds, dt=(-1, 2), only_matchup_time=True):
    """Load aviso data for multiple colocalizations and concatenate"""
    L = []
    for o in ds.obs:
        try:
            L.append(get_aviso_one_obs(ds.sel(obs=o), dt, only_matchup_time))
        except:
            assert False, o
    # ds0 = xr.concat([get_aviso_one_obs(ds.sel(obs=o), dt, only_matchup_time) for o in ds.obs], "obs")

    return xr.concat(L, "obs")


def compute_aviso_sla(ds, dt=(-1, 2), only_matchup_time=True):
    """
    https://xarray.pydata.org/en/stable/user-guide/dask.html
    Parameters
    ----------
    ds: xr.Dataset
        Input colocalization dataset
    dt: tuple
        Time offsets compared to the colocalization, e.g. if colocalization
        is at time t and dt=(-5,5), aviso data will be interpolated on
        the interval (t-5 days, t+5 days)
        Default is (-1,2)
    only_matchup_time : bool
                        if True return aviso_sla, aviso_ggx/y on the box only for the drifter matchup time
                        if False return it for all aviso time over the dt period

    Return
    ------
    ds: xr.Dataset
        Dataset containing relevant aviso data

    """
    # build template
    try:
        template = _concat_aviso(
            ds.isel(obs=slice(0, 2)), dt=dt, only_matchup_time=only_matchup_time
        )
    except:
        assert False, (ds.isel(obs=slice(0, 2)), ds.dims["obs"])

    # broad cast along obs dimension
    template, _ = xr.broadcast(
        template.isel(obs=0).chunk(),
        ds,
        exclude=[
            "aviso_time",
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
    dims = ["obs", "aviso_time", "box_y", "box_x", "site_obs"]

    template = template.transpose(
        *dims
    )  # .set_coords(["drifter_time", "drifter_x","drifter_y"])

    # actually perform the calculation
    ds_aviso = xr.map_blocks(
        _concat_aviso,
        ds,
        kwargs=dict(dt=dt, only_matchup_time=only_matchup_time),
        template=template,
    )
    ds_aviso = ds_aviso.set_coords(
        ["obs", "aviso_time_"]
    )  # coordinate for obs dimension

    """
    #add drifter_matchup
    aviso_drifter_matchup_ggx = ds_aviso.aviso_traj_ggx.isel(site_obs = ds.__site_matchup_indice.compute())
    aviso_drifter_matchup_ggy = ds_aviso.aviso_traj_ggy.isel(site_obs = ds.__site_matchup_indice.compute())
    ds_aviso['aviso_drifter_matchup_ggx'] = aviso_drifter_matchup_ggx
    ds_aviso['aviso_drifter_matchup_ggy'] = aviso_drifter_matchup_ggy
    """
    # attrs
    ds_aviso.aviso_alti_matchup_err_sla.attrs = {
        "description": "sla error interpolated on the altimeter's matchup",
        "long_name": r"$err\eta_{altimatchup}$",
        "units": "m",
    }
    ds_aviso.aviso_alti_matchup_sla.attrs = {
        "description": "sla interpolated on the altimeter's matchup",
        "long_name": r"$\eta_{altimatchup}$",
        "units": "m",
    }
    ds_aviso.aviso_alti_matchup_adt.attrs = {
        "description": "adt interpolated on the altimeter's matchup",
        "long_name": r"$adt_{altimatchup}$",
        "units": "m",
    }
    ds_aviso.aviso_alti_matchup_ggx_sla.attrs = {
        "description": "along track sla gradient term interpolated on the altimeter's matchup",
        "long_name": r"$g\partial_x\eta_{altimatchup}$",
        "units": r"$m.s^{-2}$",
    }
    ds_aviso.aviso_alti_matchup_ggy_sla.attrs = {
        "description": "cross track sla gradient term interpolated on the altimeter's matchup",
        "long_name": r"$g\partial_y\eta_{altimatchup}$",
        "units": r"$m.s^{-2}$",
    }
    ds_aviso.aviso_alti_matchup_ggx_adt.attrs = {
        "description": "along track adt gradient term interpolated on the altimeter's matchup",
        "long_name": r"$g\partial_xadt_{altimatchup}$",
        "units": r"$m.s^{-2}$",
    }
    ds_aviso.aviso_alti_matchup_ggy_adt.attrs = {
        "description": "cross track adt gradient term interpolated on the altimeter's matchup",
        "long_name": r"$g\partial_yadt_{altimatchup}$",
        "units": r"$m.s^{-2}$",
    }

    ds_aviso.aviso_box_err_sla.attrs = {
        "description": "sla error interpolated on the box at the matchup time or for several times (depending on the dimension)",
        "long_name": r"$err\eta_{box}$",
        "units": "m",
    }
    ds_aviso.aviso_box_sla.attrs = {
        "description": "sla interpolated on the box at the matchup time or for several times (depending on the dimension)",
        "long_name": r"$\eta_{box}$",
        "units": "m",
    }
    ds_aviso.aviso_box_ggx_sla.attrs = {
        "description": "along track sla gradient term interpolated on the box at the matchup time or for several times (depending on the dimension)",
        "long_name": r"$g\partial_x\eta_{box}$",
        "units": r"$m.s^{-2}$",
    }
    ds_aviso.aviso_box_ggy_sla.attrs = {
        "description": "cross track sla gradient term interpolated on the box at the matchup time or for several times (depending on the dimension)",
        "long_name": r"$g\partial_y\eta_{box}$",
        "units": r"$m.s^{-2}$",
    }
    ds_aviso.aviso_box_adt.attrs = {
        "description": "adt interpolated on the box at the matchup time or for several times (depending on the dimension)",
        "long_name": r"$adt_{box}$",
        "units": "m",
    }
    ds_aviso.aviso_box_ggx_adt.attrs = {
        "description": "along track adt gradient term interpolated on the box at the matchup time or for several times (depending on the dimension)",
        "long_name": r"$g\partial_xadt_{box}$",
        "units": r"$m.s^{-2}$",
    }
    ds_aviso.aviso_box_ggy_adt.attrs = {
        "description": "cross track adt gradient term interpolated on the box at the matchup time or for several times (depending on the dimension)",
        "long_name": r"$g\partial_yadt_{box}$",
        "units": r"$m.s^{-2}$",
    }

    ds_aviso.aviso_traj_err_sla.attrs = {
        "description": "sla error interpolated on the drifter's trajectory",
        "long_name": r"$err\eta_{traj}$",
        "units": "m",
    }
    ds_aviso.aviso_traj_sla.attrs = {
        "description": "sla interpolated on the drifter's trajectory",
        "long_name": r"$\eta_{traj}$",
        "units": "m",
    }
    ds_aviso.aviso_traj_ggx_sla.attrs = {
        "description": "along track sla gradient term interpolated on the drifter's trajectory",
        "long_name": r"$g\partial_x\eta_{traj}$",
        "units": r"$m.s^{-2}$",
    }
    ds_aviso.aviso_traj_ggy_sla.attrs = {
        "description": "cross track sla gradient term interpolated on the drifter's trajectory",
        "long_name": r"$g\partial_y\eta_{traj}$",
        "units": r"$m.s^{-2}$",
    }
    ds_aviso.aviso_traj_adt.attrs = {
        "description": "adt interpolated on the drifter's trajectory",
        "long_name": r"$adt_{traj}$",
        "units": "m",
    }
    ds_aviso.aviso_traj_ggx_adt.attrs = {
        "description": "along track adt gradient term interpolated on the drifter's trajectory",
        "long_name": r"$g\partial_xadt_{traj}$",
        "units": r"$m.s^{-2}$",
    }
    ds_aviso.aviso_traj_ggy_adt.attrs = {
        "description": "cross track adt gradient term interpolated on the drifter's trajectory",
        "long_name": r"$g\partial_yadt_{traj}$",
        "units": r"$m.s^{-2}$",
    }

    ds_aviso.aviso_drifter_matchup_ggx_sla.attrs = {
        "description": "along track sla gradient term interpolated on the drifter's matchup",
        "long_name": r"$g\partial_x\eta_{driftermatchup}$",
        "units": r"$m.s^{-2}$",
    }
    ds_aviso.aviso_drifter_matchup_ggy_sla.attrs = {
        "description": "cross track sla gradient term interpolated on the drifter's matchup",
        "long_name": r"$g\partial_y\eta_{altimetermatchup}$",
        "units": r"$m.s^{-2}$",
    }
    ds_aviso.aviso_drifter_matchup_ggx_adt.attrs = {
        "description": "along track adt gradient term interpolated on the drifter's matchup",
        "long_name": r"$g\partial_xadt_{driftermatchup}$",
        "units": r"$m.s^{-2}$",
    }
    ds_aviso.aviso_drifter_matchup_ggy_adt.attrs = {
        "description": "cross track adt gradient term interpolated on the drifter's matchup",
        "long_name": r"$g\partial_yadt_{altimetermatchup}$",
        "units": r"$m.s^{-2}$",
    }

    ds_aviso.aviso_drifter_temp_err_sla.attrs = {
        "description": "sla error interpolated on the drifter's matchup position for several times (static but temporal variation)",
        "long_name": r"$err\eta_{driftertemporal}$",
        "units": "m",
    }
    ds_aviso.aviso_drifter_temp_sla.attrs = {
        "description": "sla interpolated on the drifter's matchup position for several times (static but temporal variation)",
        "long_name": r"$\eta_{driftertemporal}$",
        "units": "m",
    }
    ds_aviso.aviso_drifter_temp_adt.attrs = {
        "description": "adt interpolated on the drifter's matchup position for several times (static but temporal variation)",
        "long_name": r"$adt_{driftertemporal}$",
        "units": "m",
    }
    return ds_aviso
