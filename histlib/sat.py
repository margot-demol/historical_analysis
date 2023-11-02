import os
from glob import glob


import xarray as xr
import pandas as pd
import numpy as np

# alti
aviso_dir = "/home/ref-cmems-public/tac/sea-level/SEALEVEL_GLO_PHY_L4_MY_008_047/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D/"

peachi_dir = "/home/datawork-cersat-public/provider/aviso/satellite/l2/sentinel-3_a/sral/peachi/sgdr/"

"""
AVISO  
---------------------------------------------------------------------------------------------------------
"""


def load_aviso(t, dt=None, suffix="", rkwargs=None, **kwargs):
    """Extract AVISO data
    Ref ...

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
    #
    D = []
    for _t in t:
        files = glob(os.path.join(aviso_dir, f"{_t.year}/{_t.dayofyear:03d}/*.nc"))
        assert len(files) == 1, f"error: multiple files at {files}"
        # ds = xr.open_dataset(files[0], **rkwargs).sel(**kwargs)
        # load_dataset loads data and close the file immediately while read_dataset is lazy
        # https://docs.xarray.dev/en/latest/generated/xarray.load_dataset.html
        # this choice may be more safe with regards to concurrent readings
        ds = xr.load_dataset(files[0], **rkwargs).sel(**kwargs)
        D.append(ds)
    ds = xr.concat(
        D, dim="time", combine_attrs="drop_conflicts"
    )  # drop attributes that vary
    ds = ds.rename(
        time=suffix + "time", longitude=suffix + "lon", latitude=suffix + "lat"
    )
    return ds


def get_aviso_one_obs(ds_obs, dt):
    """load aviso for one collocation"""

    # load data
    aviso = sat.load_aviso(ds_obs.time.values, dt, suffix="aviso_")

    # interpolate on local grid
    sla = aviso.sla.interp(
        aviso_lon=ds_obs.box_lon,
        aviso_lat=ds_obs.box_lat,
    )

    # convert to dataset and massage in order to concatenate properly in get_aviso
    ds = sla.to_dataset()
    ds["aviso_dates"] = ds.aviso_time
    ds = ds.drop(
        ["aviso_time", "time", "lon", "lat", "aviso_lon", "aviso_lat"]
    ).rename(  # forget about actual dates in time coordinate
        sla="aviso_sla"
    )

    return ds
