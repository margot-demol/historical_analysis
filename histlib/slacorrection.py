import os
from glob import glob

from xgcm import Grid
import xarray as xr
import pandas as pd
import numpy as np

import m2lib22.cstes as cstes

# alti
saral_dir = "/home/ref-cmems-public/tac/sea-level/SEALEVEL_GLO_PHY_L3_REP_OBSERVATIONS_008_062/dataset-duacs-rep-global-alg-phy-l3/"
sentinel_dir = "/home/ref-cmems-public/tac/sea-level/SEALEVEL_GLO_PHY_L3_REP_OBSERVATIONS_008_062/dataset-duacs-rep-global-s3a-phy-l3/"


"""
SLA correction  
---------------------------------------------------------------------------------------------------------
"""


def load_corr(ts, CMEMS_dir, suffix="alti_", rkwargs=None, **kwargs):
    """Extract sla correction data
    Ref ...

    Parameters
    ----------
    alti_time_list : list with time along the satelite track
    rkwargs: dict
        dictionary passed to xr.open_dataset
    **kwargs: dict
        passed to sel
    """
    if not isinstance(ts, pd._libs.tslibs.timestamps.Timestamp):
        ts = pd.to_datetime(ts)

    if rkwargs is None:
        rkwargs = {}
    # LOAD
    try:
        files = glob(
            os.path.join(
                CMEMS_dir,
                f"{ts.year}/{ts.month:02d}/*{ts.year}{ts.month:02d}{ts.day:02d}*.nc",
            )
        )
        ds = xr.load_dataset(files[0], **rkwargs).sel(**kwargs)
    except:
        assert False, ts

    ds = ds.rename(
        {"longitude": "lon", "latitude": "lat", "ocean_tide": "oceantide"}
    )  # NEW
    ds = ds.rename({l: suffix + l for l in list(ds) + list(ds.coords)})
    return ds


"""
Build dataset  
-------------------------------------------
"""


def get_corr_one_obs(ds_obs):
    """load aviso for one collocation"""

    # choose directory
    l = ds_obs.attrs["__id"]
    if "SARAL" in l:
        CMEMS_dir = saral_dir
    if "Sentinel-3_A" in l:
        CMEMS_dir = sentinel_dir

    # load data
    _drop = ["cycle", "track", "sla_filtered", "sla_unfiltered"]
    # in case time at the limit between two files
    T = ds_obs.alti_time_.values
    Tn = ds_obs.alti_time_.dropna(dim="alti_time").values  # without NaT values
    ts = pd.to_datetime(Tn[0])
    te = pd.to_datetime(Tn[-1])

    # if NaT:
    # correction series temporelle

    NaT = np.isnan(T).any()

    if NaT:
        try:
            _ds_corr = load_corr(
                ts,
                CMEMS_dir,
                suffix="alti_",
                rkwargs={"drop_variables": _drop},
                time=Tn,
                method="nearest",
                tolerance=pd.Timedelta(np.timedelta64(1, "us")),
            )
        except:
            try:
                _ds_corr = (
                    xr.concat(
                        [
                            load_corr(
                                ts,
                                CMEMS_dir,
                                suffix="alti_",
                                rkwargs={"drop_variables": _drop},
                            ),
                            load_corr(
                                ts + pd.Timedelta(np.timedelta64(1, "D")),
                                CMEMS_dir,
                                suffix="alti_",
                                rkwargs={"drop_variables": _drop},
                            ),
                        ],
                        dim="alti_time",
                    )
                    .sortby("alti_time")
                    .sel(
                        alti_time=Tn,
                        method="nearest",
                        tolerance=pd.Timedelta(np.timedelta64(1, "us")),
                    )
                )
            except:
                assert False, (ds_obs.time.values, te, ts, NaT)

        ds_nan = xr.full_like(_ds_corr.isel(alti_time=0), np.nan)
        ds_nan["alti_time"] = np.datetime64("nat", "ns")
        DS_corr = []
        for t in T:
            if np.isnat(t):
                DS_corr.append(ds_nan)
            else:
                DS_corr.append(_ds_corr.sel(alti_time=t))
        ds_corr = xr.concat(DS_corr, dim="alti_time")

    else:
        try:
            ds_corr = load_corr(
                ts,
                CMEMS_dir,
                suffix="alti_",
                rkwargs={"drop_variables": _drop},
                time=T,
                method="nearest",
                tolerance=pd.Timedelta(np.timedelta64(1, "us")),
            )
        except:
            try:
                ds_corr = (
                    xr.concat(
                        [
                            load_corr(
                                ts,
                                CMEMS_dir,
                                suffix="alti_",
                                rkwargs={"drop_variables": _drop},
                            ),
                            load_corr(
                                ts + pd.Timedelta(np.timedelta64(1, "D")),
                                CMEMS_dir,
                                suffix="alti_",
                                rkwargs={"drop_variables": _drop},
                            ),
                        ],
                        dim="alti_time",
                    )
                    .sortby("alti_time")
                    .sel(
                        alti_time=T,
                        method="nearest",
                        tolerance=pd.Timedelta(np.timedelta64(1, "us")),
                    )
                )
            except:
                assert False, (ds_obs.time.values, te, ts, NaT)

    ds_corr = ds_corr.rename({"alti_time": "alti_time_"}).rename_dims(
        {"alti_time_": "alti_time"}
    )
    ds_corr = ds_corr.assign_coords(
        alti_time=("alti_time", np.arange(ds_corr.dims["alti_time"]))
    )

    # compute sla + corrections
    ds_corr["alti_adt"] = (ds_obs.alti_sla_denoised + ds_corr.alti_mdt).assign_attrs(
        {
            "long_name": "Absolute Dynamic Topography",
            "description": "alti_sla_denoised+alti_mdt",
            "units": "m",
        }
    )
    # ds_corr['alti_sla_denoised_tide'] = (ds_obs.alti_sla_denoised + ds_corr.alti_ocean_tide).assign_attrs({'long_name': 'Sla with ocean tide', 'description':'alti_sla_denoised+alti_ocean_tide'})
    ds_corr["alti_adt_oceantide"] = (
        ds_corr.alti_adt + ds_corr.alti_oceantide
    ).assign_attrs(
        {
            "long_name": "Adt+Ocean tide",
            "description": "alti_sla_denoised+alti_mdt+alti_oceantide",
            "units": "m",
        }
    )  # NEW
    ds_corr["alti_adt_oceantide_dac"] = (
        ds_corr.alti_adt + ds_corr.alti_oceantide + ds_corr.alti_dac
    ).assign_attrs(
        {
            "long_name": "Adt+Ocean tide+dac",
            "description": "alti_sla_denoised+alti_mdt+alti_oceantide+dac",
            "units": "m",
        }
    )  # NEW

    # compute gradient
    grid = Grid(ds_obs, coords={"t": {"center": "alti_time_mid", "outer": "alti_time"}})

    dx = grid.diff(ds_obs["alti_x"], axis="t")
    dy = grid.diff(ds_obs["alti_y"], axis="t")
    dxy = np.sqrt(dx**2 + dy**2)

    g = 9.81
    for l in list(ds_corr.keys()):
        dsla = grid.diff(ds_corr[l], axis="t")
        ds_corr[l + "_g_grad_x"] = g * dsla / dxy  # NEW
        # ds_corr[l.replace('alti_', 'alti_g_grad_')] = g*dsla/dxy
    ds_corr = ds_corr.reset_coords(["alti_time_", "time"]).drop(
        ["alti_lon", "alti_lat", "alti_x", "alti_y", "lon", "lat"]
    )
    return ds_corr


def _concat_corr(ds):
    """Load sla corrections data for multiple collocations and concatenate"""
    return xr.concat([get_corr_one_obs(ds.sel(obs=o)) for o in ds.obs], "obs")


def compute_corr_sla(ds):
    """
    https://xarray.pydata.org/en/stable/user-guide/dask.html
    Parameters
    ----------
    ds: xr.Dataset
        Input collocation dataset

    Return
    ------
    ds: xr.Dataset
        Dataset containing relevant sla corrections data

    """
    template = _concat_corr(ds.isel(obs=slice(0, 2)))

    # broad cast along obs dimension
    template, _ = xr.broadcast(
        template.isel(obs=0).chunk(),
        ds,
        exclude=["box_x", "box_y", "site_obs", "alti_time", "alti_time_mid"],
    )
    # template = template.unify_chunks()
    # massage further and unify chunks
    drop_list = [
        "alti_lat",
        "alti_lon",
        "alti_time_",
        "alti_x",
        "alti_x_mid",
        "alti_y",
        "alti_y_mid",
        "box_x",
        "box_y",
        "drifter_lat",
        "drifter_lon",
        "drifter_time",
        "drifter_x",
        "drifter_y",
        "lat",
        "lon",
        "time",
    ]
    template = xr.merge(
        [template, ds.drop(drop_list)], compat="override"
    ).unify_chunks()[list(template.keys())]

    # dimension order is not consistent with that of exiting from map_blocks for some reason
    dims = ["obs", "alti_time", "alti_time_mid"]

    template = template.transpose(
        *dims
    )  # .set_coords(["drifter_time", "drifter_x","drifter_y"])

    # actually perform the calculation
    ds_corr = xr.map_blocks(_concat_corr, ds.unify_chunks(), template=template)
    ds_corr = ds_corr.set_coords(["alti_time_", "time"])  # coordinate for obs dimension

    # attrs
    ds_corr.alti_adt_g_grad_x.attrs = {
        "long_name": r"$g\partial_xadt$",
        "description": "along track absolute dynamic topography (=alti_sla_denoised+alti_mdt) gradient term ",
        "units": r"$m.s^{-2}$",
    }
    ds_corr.alti_dac_g_grad_x.attrs = {
        "long_name": r"$g\partial_xdac$",
        "description": "along track dynamic atmospheric correction gradient term ",
        "units": r"$m.s^{-2}$",
    }
    ds_corr.alti_lwe_g_grad_x.attrs = {
        "long_name": r"$g\partial_xlwe$",
        "description": "along track long wavelength error gradient term ",
        "units": r"$m.s^{-2}$",
    }
    ds_corr.alti_mdt_g_grad_x.attrs = {
        "long_name": r"$g\partial_xmdt$",
        "description": "along track mean dynamic topography gradient term ",
        "units": r"$m.s^{-2}$",
    }
    ds_corr.alti_oceantide_g_grad_x.attrs = {
        "long_name": r"$g\partial_xoceantide$",  # NEW oceantide ocean_tide
        "description": "along track ocean tide model gradient term ",
        "units": r"$m.s^{-2}$",
    }
    ds_corr.alti_adt_oceantide_g_grad_x.attrs = {
        "long_name": r"$g\partial_x(adt+oceantide)$",  # NEW
        "description": "along track adt + ocean tide gradient term ",
        "units": r"$m.s^{-2}$",
    }
    ds_corr.alti_adt_oceantide_dac_g_grad_x.attrs = {
        "long_name": r"$g\partial_x(adt+oceantide+dac)$",  # NEW
        "description": "along track adt + ocean tide + dac gradient term ",
        "units": r"$m.s^{-2}$",
    }

    return ds_corr


"""MORE ELEGANT VERSION OF GET_ONE_OBS_CORR IN PROGRESS
def get_corr_one_obs(ds_obs):
    ""load aviso for one collocation ""
    
    #choose directory
    l=ds_obs.attrs['__id']
    if 'SARAL' in l: CMEMS_dir = saral_dir
    if 'Sentinel-3_A' in l: CMEMS_dir = sentinel_dir
    
    # load data
    _drop = ['cycle','track','sla_filtered','sla_unfiltered']
    #in case time at the limit between two files
    T=ds_obs.alti_time_.values
    Tn = ds_obs.alti_time_.dropna(dim='alti_time').values # without NaT values
    ts =pd.to_datetime(Tn[0])
    te =pd.to_datetime(Tn[-1])
    
    #if NaT:
        #correction series temporelle 

    NaT=np.isnan(T).any()

                
    if NaT :        
        _ds = load_corr(ts, CMEMS_dir, suffix="alti_",
                        rkwargs={'drop_variables':_drop},
                        time =ts, method='nearest', tolerance = pd.Timedelta(np.timedelta64(1, "us")))
        ds_nan = xr.full_like(_ds, np.nan)
        ds_nan['alti_time']=np.datetime64('nat','ns')
        ds_nan['alti_lat']=np.datetime64('nat','ns')
        ds_nan['alti_lon']=np.datetime64('nat','ns') 

        try : 
            _ds_corr = xr.concat([load_corr(ts, CMEMS_dir, suffix="alti_",
                        rkwargs={'drop_variables':_drop}),ds_nan], dims='alti_time').sel(alti_time = T)
        except:
            try:
                _ds_corr = xr.concat([load_corr(ts, CMEMS_dir, suffix="alti_",rkwargs={'drop_variables':_drop}),
                                     load_corr(ts + pd.Timedelta(np.timedelta64(1, "D")), CMEMS_dir, suffix="alti_",rkwargs={'drop_variables':_drop}), ds_nan],
                                   dim="alti_time")#.sortby('alti_time').sel(alti_time = T)
                return _ds_corr
            except:
                    assert False, (ds_obs.time.values, te, ts, NaT)#, ds_obs.obs.values)
    else : 
        try : 
            ds_corr = load_corr(ts, CMEMS_dir, suffix="alti_",
                        rkwargs={'drop_variables':_drop},
                        time =T, method='nearest', tolerance = pd.Timedelta(np.timedelta64(1, "us")))
        except:
            try:
                ds_corr = xr.concat([load_corr(ts, CMEMS_dir, suffix="alti_",rkwargs={'drop_variables':_drop}),
                                     load_corr(ts + pd.Timedelta(np.timedelta64(1, "D")), CMEMS_dir, suffix="alti_",rkwargs={'drop_variables':_drop})],
                                   dim="alti_time",).sortby('alti_time').sel(alti_time=T, method='nearest', tolerance = pd.Timedelta(np.timedelta64(1, "us")))
            except:
                assert False, (ds_obs.time.values, te, ts, NaT)#, ds_obs.obs.values)
                        
        
        

            
    ds_corr = ds_corr.rename({'alti_time':'alti_time_'}).rename_dims({'alti_time_':'alti_time'})
    ds_corr = ds_corr.assign_coords(alti_time=("alti_time", np.arange(ds_corr.dims['alti_time'])))
    
    #compute sla + corrections    
    ds_corr['alti_adt']= (ds_obs.alti_sla_denoised + ds_corr.alti_mdt).assign_attrs({'long_name': 'Absolute Dynamic Topography', 'description':'alti_sla_denoised+alti_mdt'})
    #ds_corr['alti_sla_denoised_tide'] = (ds_obs.alti_sla_denoised + ds_corr.alti_ocean_tide).assign_attrs({'long_name': 'Sla with ocean tide', 'description':'alti_sla_denoised+alti_ocean_tide'})
        

    
    #compute gradient 
    grid = Grid(ds_obs, coords={"t": {"center": "alti_time_mid", "outer": "alti_time"}})
    
    dx = grid.diff(ds_obs["alti_x"], axis="t")
    dy = grid.diff(ds_obs["alti_y"], axis="t")
    dxy = np.sqrt(dx**2+dy**2)

    
    g = 9.81
    for l in list(ds_corr.keys()) :
        dsla = grid.diff(ds_corr[l], axis="t")
        ds_corr[l.replace('alti_', 'alti_g_grad_')] = g*dsla/dxy
    ds_corr = ds_corr.reset_coords(["alti_time_", 'time']).drop(['alti_lon', 'alti_lat','alti_x', 'alti_y','lon','lat'])
    return _ds_corr
"""
