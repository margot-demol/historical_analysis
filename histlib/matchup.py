import numpy as np
import pandas as pd
import xarray as xr
import pyproj
from rasterio.transform import Affine
from math import ceil

import matplotlib.pyplot as plt
import matplotlib.colors as cl

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

import histlib.box as box
#import histlib.sat as sat
import histlib.cstes as cstes
import histlib.stress_to_windterm as stw

from histlib.stress_to_windterm import list_wd_srce_suffix, list_func, list_func_suffix
from histlib.cstes import labels, zarr_dir



""" 
------------------------------------------------------------------------
MATCHUP DATASET
------------------------------------------------------------------------
"""

def add_ggx_attrs(ds_data):
    """Add gradient attributes
    Parameters
    ----------
    ds_data: dataset
            dataset containing colocalisations
    """
    listv = [l for l in list(ds_data.variables) if 'sla' in l]+['alti_mdt','alti_ocean_tide', 'alti_dac', 'alti_internal_tide']
    listv = [l for l in listv if 'gg' not in l]
    for v in listv :
        ds_data[v.replace('alti', 'alti_ggx')].attrs['comment'] = ds_data[v].attrs['comment']
        ds_data[v.replace('alti', 'alti_ggx')].attrs['units'] = r'$m.s^{-2}$'
        ds_data[v.replace('alti', 'alti_ggx')].attrs['long_name']= r'$g\partial_x$'+v.replace('alti_','')

def add_adt_to_ds_data(ds_data):
    """Adds adt to dataset from sla + mdt and manages attributes
    Parameters
    ----------
    ds_data: dataset
            dataset containing colocalisations
    """
    add_ggx_attrs(ds_data)
    ds_data = ds_data.rename({'drifter_acc_x':'drifter_acc_x_0', 'drifter_acc_y':'drifter_acc_y_0', 'drifter_coriolis_x':'drifter_coriolis_x_0', 'drifter_coriolis_y':'drifter_coriolis_y_0'})
    for sla in ['alti_ggx_sla_filtered','alti_ggx_sla_unfiltered','alti_ggx_sla_unfiltered_denoised','alti_ggx_sla_unfiltered_imf1']:
        ds_data[sla.replace('sla', 'adt')] = ds_data[sla] + ds_data.alti_ggx_mdt
        ds_data[sla.replace('sla', 'adt')].attrs['comment'] = ds_data[sla].attrs['comment']
        ds_data[sla.replace('sla', 'adt')].attrs['units'] = r'$m.s^{-2}$'
        ds_data[sla.replace('sla', 'adt')].attrs['long_name']= ds_data[sla].attrs['long_name'].replace('sla', 'adt')
    return ds_data

def change_obs_coords(ds,l) :
    """Change obs coordinates including the label
    Parameters
    ----------
    ds_data: dataset
            dataset containing colocalisations
    l : label, 
            in cstes.labels
    """
    o = np.array(ds['obs']).astype('U')
    L = np.full_like(o, l+'__')
    ob = xr.DataArray(np.char.add(L, o), dims='obs')
    ds = ds.drop('obs')
    ds = ds.assign_coords({'obs':ob})
    return ds
    
import histlib.stress_to_windterm as stw
_data_var = [
    "f",
    "box_theta_lon",
    "__site_matchup_indice",
    "box_theta_lat",
    "drifter_theta_lon",
    "drifter_theta_lat",
    "drifter_typebuoy",
    "alti___distance",
    "alti___time_difference",
    'alti_ggx_dac',
    'alti_ggx_internal_tide',
    'alti_ggx_mdt',
    'alti_ggx_ocean_tide',
    'alti_ggx_sla_filtered',
    'alti_ggx_sla_unfiltered',
    'alti_ggx_sla_unfiltered_denoised',
    #'alti_ggx_sla_unfiltered_imf1',
    'alti_ggx_adt_filtered',
    'alti_ggx_adt_unfiltered',
    'alti_ggx_adt_unfiltered_denoised',
    #'alti_ggx_adt_unfiltered_imf1',
    "drifter_vx",
    "drifter_vy",
    "drifter_acc_x_0",
    "drifter_acc_y_0",
    "drifter_coriolis_x_0",
    "drifter_coriolis_y_0",
    
]
_aviso_var = [
    "aviso_alti_matchup_ggx_adt",
    "aviso_alti_matchup_ggy_adt",
    "aviso_drifter_matchup_ggx_adt",
    "aviso_drifter_matchup_ggy_adt",
    "aviso_alti_matchup_ggx_sla",
    "aviso_alti_matchup_ggy_sla",
    "aviso_drifter_matchup_ggx_sla",
    "aviso_drifter_matchup_ggy_sla",
]
_stress_var = [
    "e5_alti_matchup_taue",
    "e5_alti_matchup_taun",
    "es_alti_matchup_taue",
    "es_alti_matchup_taun",
    "e5_drifter_matchup_taue",
    "e5_drifter_matchup_taun",
    "es_drifter_matchup_taue",
    "es_drifter_matchup_taun",
]


list_wd_srce_suffix = ["es", "e5"]
list_func = [stw.cst_rio_z0, stw.cst_rio_z15]
list_func_suffix = ["cstrio_z0", "cstrio_z15"]


""" 
------------------------------------------------------------------------
MATCHUP FILE COLOC + AVISO + ERASTAR
------------------------------------------------------------------------
"""
def matchup_dataset_one(l, T=10, cutoff=[2,1,0.5,0.2, 0.1]):
    """Gathers colocalisation, erastar and aviso data at matchup point, include drogue status. Return the matchup dataset corresponding to l
    Parameters
    ----------
    l : label, 
            in cstes.labels
    T : window length of low pass filter on drifter trajectories (days)
    cutoff : cutoff frequencies (cpd)
    """
    ds_data = xr.open_zarr(os.path.join(zarr_dir, f'{l}.zarr')).chunk({'obs':500, 'site_obs':-1})
    ds_data = ds_data.where(ds_data.alti___distance<2e5, drop=True).chunk({'obs':500, 'site_obs':-1})
    ds_data = add_adt_to_ds_data(ds_data)

    drogue_status = ds_data.time<ds_data.drifter_drogue_lost_date.mean('site_obs')
    ds_data = ds_data[_data_var]
    
    if cutoff!=None:
        add_low_pass_filter_to_data(ds_data, T=10, cutoff=[2,1,0.5,0.2, 0.1])
    
    ds_aviso = xr.open_zarr(os.path.join(zarr_dir+'_ok','aviso', f'aviso_{l}.zarr')).chunk({'obs':500})
    ds_stress = xr.open_zarr(os.path.join(zarr_dir+'_ok','erastar', f'erastar_{l}.zarr')).chunk({'obs':500})
    
    # COLOCALIZATIONS DATA
    cc = [l for l in list(ds_data.coords)if l not in ['obs', 'box_x', 'box_y', 'alti_time_mid']]
    ds_data = ds_data.reset_coords(cc)
    idx = ds_data.__site_matchup_indice.astype(int).compute()
    _ds_data = ds_data.sel(site_obs=idx).isel( alti_time_mid=ds_data.dims['alti_time_mid']//2).drop(["alti_time_mid", "alti_x_mid", "alti_y_mid"])
    #_ds_data = compute_dsdatamatchup(ds_data)
    for v in _ds_data.variables:
        _ds_data[v].attrs = ds_data[v].attrs
    _ds_data['drogue_status'] = drogue_status.drop(['lon', 'lat', 'time']).assign_attrs({'long_name':'drogue status', 'description':'True if drogued, False if undrogued (day precision only)'})

    # AVISO
    _ds_aviso = ds_aviso[_aviso_var]

    #ERASTAR
    _ds_stress = ds_stress[_stress_var]

    # FOR IND PDFS
    _ds = xr.merge([_ds_data, _ds_aviso, _ds_stress])

    # COMPUTE WD TERM
    _ds = xr.merge(
        [
            _ds,
            stw.compute_wd_from_stress(
                _ds, list_wd_srce_suffix, list_func, list_func_suffix, False
            ),
        ]
    )
    # CLEANING : drop useless variables
    _ds = _ds.drop(
        _stress_var
        + [
            "f",
            "box_theta_lon",
            "box_theta_lat",
            "drifter_theta_lon",
            "drifter_theta_lat",
            "__site_matchup_indice",
        ]
    ).set_coords(["alti___distance", "alti___time_difference"])
    _ds = _ds.rename({v: v.replace("_matchup", "") for v in _ds})
    #_ds = change_obs_coords(_ds, l)  
    _ds = _ds.drop(['box_x', 'box_y']).set_coords(['lon', 'lat','time'])

    #ADDING TIDE + DAC correction
    _ds['alti_ggx_adt_filtered_ocean_tide'] = _ds.alti_ggx_adt_filtered+ _ds.alti_ggx_ocean_tide
    _ds['alti_ggx_adt_filtered_ocean_tide_internal_tide'] = _ds.alti_ggx_adt_filtered_ocean_tide + _ds.alti_ggx_internal_tide
    _ds['alti_ggx_adt_filtered_ocean_tide_internal_tide_dac'] = _ds.alti_ggx_adt_filtered_ocean_tide_internal_tide + _ds.alti_ggx_internal_tide

    _ds['alti_ggx_adt_filtered_ocean_tide'].attrs=_ds.alti_ggx_adt_filtered.attrs
    _ds['alti_ggx_adt_filtered_ocean_tide_internal_tide'].attrs = _ds.alti_ggx_adt_filtered.attrs
    _ds['alti_ggx_adt_filtered_ocean_tide_internal_tide_dac'].attrs = _ds.alti_ggx_adt_filtered.attrs


    return _ds

"""
------------------
ADD LOW PASS FILTER
------------------
"""
def add_low_pass_filter_to_data(ds, T=10, cutoff=[2,1,0.5,0.2, 0.1]) : 
    """ Return dataset with filtered trajectories acceleration and coriolis term 
        T: window length days
        cutoff : cut off frequency in cpd
    """
    from scipy.signal import filtfilt
    from scipy.integrate import cumulative_trapezoid
    from scipy.optimize import minimize
    # coefficients
    dt = (ds.drifter_time.diff('site_obs')/pd.Timedelta('1D')).mean()  # in days
    from pynsitu.tseries import generate_filter

    dss = ds[[ 'drifter_vx', 'drifter_vy']].compute().fillna(0)
    if not isinstance(cutoff, list) : cutoff=[cutoff]
    for cto in cutoff :
        taps = generate_filter(band="low", dt=dt, T=T, bandwidth=cto)
        print(len(taps))
        try : 
            vx = xr.DataArray(filtfilt(taps, 1, dss.drifter_vx),dims=['obs', 'site_obs'])
            vy = xr.DataArray(filtfilt(taps, 1, dss.drifter_vy), dims=['obs', 'site_obs'])
        except :
            print('padlen modified')
            vx = xr.DataArray(filtfilt(taps, 1, dss.drifter_vx, padlen = len(dss.drifter_vx)),dims=['obs', 'site_obs'])
            vy = xr.DataArray(filtfilt(taps, 1, dss.drifter_vy, padlen = len(dss.drifter_vx)), dims=['obs', 'site_obs'])
            
        cutoffstr = str(cto).replace('.','')

        
        
        ds[f"drifter_acc_x_{cutoffstr}"] = (vx.differentiate("site_obs")/3600).assign_attrs(**ds.drifter_acc_x_0.attrs).assign_attrs(description= ds.drifter_acc_x_0.attrs['description'] + f' filtered with {cto} cpd frequency',cutoff=cto)
        ds[f"drifter_acc_y_{cutoffstr}"] = (vy.differentiate("site_obs")/3600).assign_attrs(**ds.drifter_acc_x_0.attrs).assign_attrs(description= ds.drifter_acc_y_0.attrs['description'] + f' filtered with {cto} cpd frequency',cutoff=cto)
        ds[f"drifter_coriolis_x_{cutoffstr}"] = (-vy * ds.f).assign_attrs(ds.drifter_coriolis_x_0.attrs).assign_attrs(description= ds.drifter_coriolis_x_0.attrs['description'] + f' filtered with {cto} cpd frequency',cutoff=cto)
        ds[f"drifter_coriolis_y_{cutoffstr}"] = (vx * ds.f).assign_attrs(ds.drifter_coriolis_y_0.attrs).assign_attrs(description= ds.drifter_coriolis_y_0.attrs['description'] + f' filtered with {cto} cpd frequency',cutoff=cto)


    
"""
------------------
COLOCALISATIONS
------------------
"""
def find_term_list(_ds, wd_x=None, wd_y=None, grad_x=None, grad_y=None, cutoff=None):
    """ Find lists of every possibilities for each momentum equation terms from the dataset
    Parameters
    ----------
    _ds: dataset
            dataset containing colocalisations
    wd_x, wd_y, grad_x, grad_y : list of str
        contains wd_x, wd_y, grad_x or grad_y variable names in the dataset,
        if not provided, will find all possibly corresponding variables in dataset
    cutoff : list of str
            cutoff frequency for drifter variables ('0' for no filter applied)
    
    """
    if not wd_x : wd_x = [l for l in _ds if "wd_x" in l]
    if not wd_y : wd_y = [l for l in _ds if "wd_y" in l]
    if not grad_x : grad_x = [l for l in _ds if "ggx_adt" in l]# or "ggx_sla" in l]
    if not grad_y : grad_y = [l for l in _ds if "ggy_adt" in l]# or "ggy_sla" in l]
    if not cutoff : cutoff = [l.split('acc_x_')[-1] for l in _ds if "acc_x" in l]
    return wd_x, wd_y, grad_x, grad_y, cutoff
    
def combinations(_ds, wd_x=None, wd_y=None, grad_x=None, grad_y=None, cutoff=None):
    """Create a list of dictionnaries containing the different data combinations possible to rebuild the moment conservation

    Parameters
    ----------
    _ds: dataset
        contains - drifter_acc_x/y,
                 - drifter_coriolisx/y,
                 - sla gradients from the different sources all fiishing with '_ggx/y',
                 - wind terms from the different sources and way to compute it from stress all finishing with '_wd_x/y'
    wd_x, wd_y, grad_x, grad_y : list of str
        contains wd_x, wd_y, grad_x or grad_y variable names in the dataset we want to consider for combinations
    cutoff : list of str
            cutoff frequency for drifter variables ('0' for no filter applied) we want to consider for combinations
    Returns
    ----------
    [{'acc': 'drifter_acc_x','coriolis': 'drifter_coriolis_x','ggx': 'alti_ggx','wind': 'es_cstrio_z0_alti_wd_x','id': 'co_es_cstrio_z0_alti_x'},....]
    list of dictionnaries containing the varaibles taken for each term and an identification: gradsrc_wdsrc_wdmethod_wddepth_matchupposition_x/y
    """
    
    wd_x, wd_y, grad_x, grad_y, cutoff = find_term_list(_ds, wd_x, wd_y, grad_x, grad_y, cutoff)
    
    LIST = []

    for cf in cutoff :
        for grad in grad_x :
            for wd in wd_x :
                lx = {
                    "acc": 'drifter_acc_x_'+cf,
                    "coriolis": 'drifter_coriolis_x_'+cf,
                    "ggrad": "",
                    "wind": "",
                    "id": "",
                }
                # AVISO grad
                if "aviso" in grad:  #
                    if ("alti" in grad) and ("alti" in wd):
                        lx["ggrad"] = grad
                        lx["wind"] = wd
                        lx["id"] = (
                            "aviso__"
                            +cf+'__'
                            +grad[-3:]
                            +'__'
                            +"_".join(wd.split("_")[:3])
                            + "__alti_x"
                        )
                        LIST.append(lx)
                    elif ("drifter" in grad) and ("drifter" in wd):
                        lx["ggrad"] = grad
                        lx["wind"] = wd
                        lx["id"] = (
                            "aviso__"
                            +cf+'__'
                            +grad[-3:]
                            +'__'
                            + "_".join(wd.split("_")[:3])
                            + "__drifter_x"
                        )
                        LIST.append(lx)
                # Altimeters' grad
                elif "alti" in grad:
                    if "alti" in wd:
                        lx["ggrad"] = grad
                        lx["wind"] = wd
                        lx["id"] = (
                            "co__"
                            +cf+'_'
                            + grad.replace("alti_", "").replace("ggx", "")
                            +'__'
                            + "_".join(wd.split("_")[:3])
                            + "__alti_x"
                        )
                        LIST.append(lx)
                    elif "drifter" in wd:
                        lx["ggrad"] = grad
                        lx["wind"] = wd
                        lx["id"] = (
                            "co__"
                            +cf+'_'
                            + grad.replace("alti_", "").replace("ggx", "")
                            +'__'
                            + "_".join(wd.split("_")[:3])
                            + "__drifter_x"
                        )
                        LIST.append(lx)
        for grad in grad_y:
            for wd in wd_y:
                ly = {
                    "acc": "drifter_acc_y_"+cf,
                    "coriolis": "drifter_coriolis_y_"+cf,
                    "ggrad": "",
                    "wind": "",
                    "id": "",
                }
    
                if ("alti" in grad) and ("alti" in wd):
                    ly["ggrad"] = grad
                    ly["wind"] = wd
                    ly["id"] = (
                        "aviso__"
                        +cf+'__'
                        +grad[-3:]
                        + "__"
                        + "_".join(wd.split("_")[:3])
                        + "__alti_y"
                    )
                    LIST.append(ly)
    
                elif ("drifter" in grad) and ("drifter" in wd):
                    ly["ggrad"] = grad
                    ly["wind"] = wd
                    ly["id"] = (                            
                        "aviso__"
                        +cf+'__'
                        +grad[-3:]
                        + "__"
                        + "_".join(wd.split("_")[:3])
                        + "__drifter_y"
                    )
                    LIST.append(ly)
    return LIST
"""
------------------
ADD EXCEPT + SUM
------------------
"""
def add_except_sum(
    ds_matchup,
    sum_= True,
    except_ = True,
    wd_x=None,
    wd_y=None,
    grad_x=None,
    grad_y=None,
    cutoff=None
):
    """Create dataset with matchup terms values, sum of terms and except sum of terms for all possible combinations
    

    Parameters
    ----------
    ds_matchup: dataset
            dataset containing colocalisations, should contain at least the _data_var
    _sum : bool
            add sum terms or not, default is True
    _except : bool
            add except terms or not, default is True
    wd_x, wd_y, grad_x, grad_y : list of str
        contains wd_x, wd_y, grad_x or grad_y variable names in the dataset we want to consider for combinations
        if not provided, will find all possibly corresponding variables in dataset
    cutoff : list of str
            cutoff frequency for drifter variables ('0' for no filter applied) we want to consider for combinations`
            if not provided, will find all possibly corresponding variables in dataset
    """
    
    # SUM combination
    wd_x, wd_y, grad_x, grad_y, cutoff = find_term_list(ds_matchup, wd_x, wd_y, grad_x, grad_y, cutoff)
        
    COMB = combinations(ds_matchup, wd_x, wd_y, grad_x, grad_y, cutoff)
    ds_matchup = ds_matchup[wd_x + wd_y+ grad_x + grad_y + ["drifter_acc_x_"+cf for cf in cutoff]+["drifter_acc_y_"+cf for cf in cutoff]+["drifter_coriolis_x_"+cf for cf in cutoff]+["drifter_coriolis_y_"+cf for cf in cutoff]]
    _ds_sum = xr.Dataset()
    _ds_except = xr.Dataset()

    id_comb_list = []
    for comb in COMB:
        #print(comb.values())
        _id = comb["id"]
        id_comb_list.append(_id)
        comb.pop("id")
        ds_matchup["id_comb"] = id_comb_list

        if sum_:
            # TOTAL SUM
            S = 0

            for l in list(comb.values()):
                S = S + ds_matchup[l]
            id_str = "sum_" + _id
            _ds_sum[id_str] = xr.DataArray(
                data=S,
                attrs={
                    "description": "+".join(comb.keys()),
                    "long_name": "+".join(
                        [ds_matchup[comb[v]].attrs["long_name"] for v in comb.keys()]
                    ),
                    "units": r"$m.s^{-2}$",
                    **comb,
                },
            )
            _ds_sum["id_comb"] = id_comb_list

        if except_:
            # EXCEPT ONE
            for except_key in list(comb.keys()):
                id_str_2 = "exc_" + except_key + "_" + _id
                S2 = 0
                keys = [l for l in comb.keys() if l != except_key]
                for key in keys:
                    if key != except_key:
                        S2 = S2 + ds_matchup[comb[key]]
                _ds_except[id_str_2] = xr.DataArray(
                    data=S2,
                    attrs={
                        "description": "+".join(keys),
                        "long_name": "+".join(
                            [ds_matchup[comb[v]].attrs["long_name"] for v in keys]
                        ),
                        "units": r"$m.s^{-2}$",
                        **comb,
                    },
                )
                _ds_except["id_comb"] = id_comb_list

    DS = [ds_matchup]
    if except_:
        DS.append(_ds_except)
    if sum_:
        DS.append(_ds_sum)
    return xr.merge(DS)

def terms_comb(id_):
    l = id_.split('__')
    xy = id_[-1]
    attrs = {'acc':'drifter_acc_'+xy+'_'+l[1],'coriolis': 'drifter_coriolis_'+xy+'_'+l[1]}
    if 'co' in l:
        attrs['gg'] = 'alti_gg'+xy+'_'+l[2]
    if 'aviso' in l :
        attrs['gg'] = 'aviso_'+l[4][:-3]+'_gg'+xy+'_'+l[2]
    attrs['wd'] = l[3]+'_'+l[4].replace('_', '_wd_')
    return attrs

"""
Diagnosis functions
-------------------------------
"""
dl = 1
lon, lat = np.arange(-180, 180, dl), np.arange(-90, 90, dl)
bins_lonlat = [[lon, lat]]


def coloc_repartition(ds, bins=bins_lonlat):
    """Count number of colocalisation in lon-lat bins and add it to the diagnosis dataset

    Parameters
    ----------
    ds : xarray Dataset
        dataset build by box.build_dataset
    bins : [np.array,np.array]
        [lon,lat]
    ds_diag : xarray Dataset
        diagnosis dataset in which we add new diagnosis
    """

    return histogram(ds.lon, ds.lat, bins=bins).compute()


def compute_pdfs(ds, bins):
    """Compute and add to the diagnosis dataset independant pdf of the acceleration's, coriolis's terms for both x-along track and y-orthogonal direction and pressure gradient's term on both x-along track
    If the given bin is np.ndarray, it returns the pdfs of the flatten ds, if the given bin is a dictionnary, the pdfs will be computed along dict dimensions.

    Parameters
    ----------
    ds : xarray Dataset
        dataset build by box.build_dataset
    bins : np.ndarray or dict('dim':ndarray)
    ds_diag : xarray Dataset
        diagnosis dataset in which we add new diagnosis
    """
    ds_diag = xr.Dataset()
    list_var = list(ds.keys())
    # bins = dict(acc=bins_acc, separation=bins_separation)

    # Acceleration
    dt = 3600
    if isinstance(bins, dict):
        for var in list_var:
            ds_diag["pdf_" + var] = (
                histogram(
                    ds[var].rename("acc"),
                    *[ds[k] for k in bins if k != "acc"],
                    bins=[v for k, v in bins.items()],
                    density=True
                )
                .assign_attrs(
                    {
                        **ds[var].attrs,
                        "description": "independant pdf of "
                        + ds[var].attrs["description"],
                        "long_name": "pdf[" + ds[var].attrs["long_name"] + "]",
                        "units": "",
                    }
                )
                .compute()
            )

    elif isinstance(bins, np.ndarray):
        for var in list_var:
            ds_diag["pdf_" + var] = (
                histogram(ds[var].rename("acc"), bins=bins, density=True)
                .assign_attrs(
                    {
                        **ds[var].attrs,
                        "description": "independant pdf of "
                        + ds[var].attrs["description"],
                        "long_name": "pdf[" + ds[var].attrs["long_name"] + "]",
                        "units": "",
                    }
                )
                .compute()
            )
    else:
        print("wrong bins type")

    ds_diag["nb_coloc"] = ds.sizes["obs"]
    if "id_comb" in ds:
        ds_diag["id_comb"] = ds["id_comb"]

    return ds_diag


def ds_mean_var_std(
    ds_pdf, bin_dim, mean=False, rms=False, var=False, std=True, ms=True
):
    """Return dataset containing mean, variance, rms and std for all pdfs in the given dataset

    Parameters
    ----------
    ds_pdf : xarray Dataset
        dataset containing pdfs built by diags
    bin_dim : str
        bin dimension name
    """

    list_pdf_name = [l.replace("pdf", "") for l in list(ds_pdf.keys()) if "pdf" in l]
    _ds = xr.Dataset()

    for name in list_pdf_name:
        rescale = ds_pdf["pdf" + name].integrate(coord=bin_dim)

        _ds["mean" + name] = (
            (ds_pdf[bin_dim] * ds_pdf["pdf" + name]).integrate(coord=bin_dim) / rescale
        ).assign_attrs(
            {
                **ds_pdf["pdf" + name].attrs,
                "description": ds_pdf["pdf" + name]
                .attrs["description"]
                .replace("independant pdf of ", "mean of "),
                "long_name": r"$\langle$"
                + ds_pdf["pdf" + name].attrs["long_name"].split("[")[1].replace("]", "")
                + r"$\rangle$",
                "units": r"$m.s^{-2}$",
            }
        )

        _ds["var" + name] = (
            (
                (ds_pdf[bin_dim] - _ds["mean" + name]) ** 2 * ds_pdf["pdf" + name]
            ).integrate(coord=bin_dim)
            / rescale
        ).assign_attrs(
            {
                **ds_pdf["pdf" + name].attrs,
                "description": ds_pdf["pdf" + name]
                .attrs["description"]
                .replace("independant pdf of ", "variance of "),
                "long_name": ds_pdf["pdf" + name]
                .attrs["long_name"]
                .replace("pdf", "var"),
                "units": r"$m^2.s^{-4}$",
            }
        )

        _ds["std" + name] = np.sqrt(_ds["var" + name]).assign_attrs(
            {
                **ds_pdf["pdf" + name].attrs,
                "description": ds_pdf["pdf" + name]
                .attrs["description"]
                .replace("independant pdf of ", "standard deviation of "),
                "long_name": ds_pdf["pdf" + name]
                .attrs["long_name"]
                .replace("pdf", r"$\sigma$"),
                "units": r"$m.s^{-2}$",
            }
        )

        _ds["rms" + name] = np.sqrt(
            (np.square(ds_pdf[bin_dim]) * ds_pdf["pdf" + name]).integrate(coord=bin_dim)
            / rescale
        ).assign_attrs(
            {
                **ds_pdf["pdf" + name].attrs,
                "description": ds_pdf["pdf" + name]
                .attrs["description"]
                .replace("independant pdf of ", "root mean square of "),
                "long_name": ds_pdf["pdf" + name]
                .attrs["long_name"]
                .replace("pdf", "rms"),
                "units": r"$m^2.s^{-4}$",
            }
        )

        _ds["ms" + name] = (
            (np.square(ds_pdf[bin_dim]) * ds_pdf["pdf" + name]).integrate(coord=bin_dim)
            / rescale
        ).assign_attrs(
            {
                **ds_pdf["pdf" + name].attrs,
                "description": ds_pdf["pdf" + name]
                .attrs["description"]
                .replace("independant pdf of ", "mean square of "),
                "long_name": ds_pdf["pdf" + name]
                .attrs["long_name"]
                .replace("pdf", "ms"),
                "units": r"$m.s^{-2}$",
            }
        )

    if not mean:
        _ds = _ds.drop([v for v in _ds.keys() if "mean" in v])
    if not var:
        _ds = _ds.drop([v for v in _ds.keys() if "var" in v])
    if not std:
        _ds = _ds.drop([v for v in _ds.keys() if "std" in v])
    if not rms:
        _ds = _ds.drop([v for v in _ds.keys() if "rms" in v])
    if not ms:
        _ds = _ds.drop([v for v in _ds.keys() if "ms" in v])

    if "id_comb" in ds_pdf:
        _ds["id_comb"] = ds_pdf["id_comb"]

    if "nb_coloc_bins" in ds_pdf:
        _ds["nb_coloc_bins"] = ds_pdf["nb_coloc_bins"]

    return _ds


def ds_mean_var_std_2bins(ds, bin_dim, bin_dim_ggrad, **kwargs):
    return xr.merge(
        [
            ds_mean_var_std(
                ds[[var for var in ds if var != "pdf_alti_ggx"]],
                bin_dim="acc_bin",
                **kwargs
            ),
            ds_mean_var_std(
                ds[["pdf_alti_ggx"]], bin_dim="acc_bin_grad", **kwargs
            ),
        ]
    )


def global_pdf(ds_pdf, min_coloc_bin=50, **kwargs):
    ds_pdf_all = (ds_pdf.nb_coloc * ds_pdf[[v for v in ds_pdf if "pdf" in v]]).sel(
        **kwargs
    ).sum("drifter_sat_year") / (
        ds_pdf.nb_coloc.sel(**kwargs).sum("drifter_sat_year")
    )  # normalized
    if "nb_coloc_bins" in ds_pdf:
        ds_pdf_all["nb_coloc_bins"] = ds_pdf.nb_coloc_bins.sum("drifter_sat_year")
        ds_pdf_all = ds_pdf_all.where(ds_pdf_all.nb_coloc_bins >= min_coloc_bin)
    if "id_comb" in ds_pdf:
        ds_pdf_all["id_comb"] = ds_pdf.id_comb
    for v in ds_pdf_all:
        ds_pdf_all[v].attrs = ds_pdf[v].attrs
    return ds_pdf_all


"""
PLOT
"""


def plot_stat_lonlat(
    variables, ds=1, cmap="viridis", title=1, cmap_label=1, fig_title=1
):
    lv = len(variables)
    if isinstance(variables[0], str):
        variables = [ds[v] for v in variables]
        if ds == 1:
            assert False, "give dataset"
    nrows = ceil(lv / 2)
    ncols = 2
    if lv == 1:
        nrows = 1
        ncols = 1
    # Define the figure and each axis for the 3 rows and 3 columns
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(16, nrows * 4),
    )

    # axs is a 2 dimensional array of `GeoAxes`.  We will flatten it into a 1-D array
    if lv != 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    # Loop over all of the variables
    for i in range(lv):
        # Contour plot
        if cmap_label != 1:
            variables[i].assign_attrs({"long_name": cmap_label[i]}).plot(
                x="lon_bin", y="lat_bin", cmap=cmap, ax=axs[i]
            )
        else:
            variables[i].plot(x="lon_bin", y="lat_bin", cmap=cmap, ax=axs[i])

        # Title each subplot with the name of the model
        if title != 1:
            axs[i].set_title(title[i], fontsize=15)

        # Draw the coastines for each subplot
        axs[i].coastlines()
        axs[i].add_feature(cfeature.LAND)
        gl = axs[i].gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=1,
            color="gray",
            alpha=0.5,
            linestyle=":",
        )
        gl.xlabels_top = False
        gl.ylabels_right = False
    if isinstance(fig_title, str):
        fig.suptitle(fig_title)
    fig.tight_layout(rect=[0, 0, 1, 1])  # left, bottom, right, top (default is 0,0,1,1)
    # set the spacing between subplots

    # Delete the unwanted axes
    if lv != 1:
        for i in np.arange(lv, nrows * 2):
            fig.delaxes(axs[i])


def plot_ms_lonlat(ds, id_, title=1):
    dic = ds["ms_sum_" + id_].attrs
    acc = "ms_" + dic["acc"]
    cor = "ms_" + dic["coriolis"]
    ggrad = "ms_" + dic["ggrad"]
    wd = "ms_" + dic["wind"]

    ggrad_tick = (
        ds[ggrad]
        .attrs["long_name"]
        .replace("ms[", r"")
        .replace("]", "")
        .replace("altimatchup", "aviso")
        .replace("driftermatchup", "aviso")
    )
    w = ds[wd].attrs["long_name"].replace("ms[", r"").replace("]", "").split(" from")
    wd_tick = " from".join(w)
    ticks = [r"$d_tu$", r"$-fv$", ggrad_tick, wd_tick]

    # S
    S = ds["ms_sum_" + id_]
    title = [r"$\langle S^2\rangle$"]
    plot_stat_lonlat([S], title=title, cmap_label=title, fig_title=id_)

    # S-x
    Sx = [
        ds["ms_exc_acc_" + id_] / S,
        ds["ms_exc_coriolis_" + id_] / S,
        ds["ms_exc_ggrad_" + id_] / S,
        ds["ms_exc_wind_" + id_] / S,
    ]
    title = [
        r"$\frac{\langle S_{-x}^2\rangle}{\langle S^2\rangle}$   $x =$" + t
        for t in ticks
    ]
    cmap_label = [r"$\langle S_{-x}^2\rangle/\langle S^2\rangle$"] * len(Sx)
    plot_stat_lonlat(Sx, title=title, cmap_label=cmap_label, fig_title=id_)

    # x
    x = [ds[acc] / S, ds[cor] / S, ds[ggrad] / S, ds[wd] / S]
    title = [
        r"$\frac{\langle x^2\rangle}{\langle S^2\rangle}$   $x =$" + t for t in ticks
    ]
    cmap_label = [r"$\langle x^2\rangle/\langle S^2\rangle$"] * len(Sx)
    plot_stat_lonlat(x, title=title, cmap_label=cmap_label, fig_title=id_)

""" 
------------------------------------------------------------------------
PB selection solved, unused
------------------------------------------------------------------------

def get_dsdatamatchup_one_obs(ds):
    return ds.isel(site_obs=int(ds.__site_matchup_indice.values), alti_time_mid=ds.dims['alti_time_mid']//2).drop(["alti_time_mid", "alti_x_mid", "alti_y_mid"])
    # CAUTION __site_matchup_indice
def _concat_dsdatamatchup(ds):
    L=[]
    for o in ds.obs :
        try : 
            L.append(get_dsdatamatchup_one_obs(ds.sel(obs=o)))
        except :
            assert False, o
    return xr.concat(L, "obs")


def compute_dsdatamatchup(ds):

    # build template
    try : 
        template = _concat_dsdatamatchup(ds.isel(obs=slice(0, 2)).compute())
    except : assert False, (ds.isel(obs=slice(0, 2)), ds.dims['obs'])

    # broad cast along obs dimension
    template, _ = xr.broadcast(
        template.isel(obs=0).chunk(),
        ds,
        exclude=[
            "alti_time_mid",
            "site_obs",
            "box_y",
            "box_x",
        ],
    )

    template = template.chunk({'obs':500})

    # dimension order is not consistent with that of exiting from map_blocks for some reason
    dims = ["obs", 'box_y', 'box_x']

    template = template.transpose(
        *dims
    )  # .set_coords(["drifter_time", "drifter_x","drifter_y"])

    # actually perform the calculation
    dsdatamatchup = xr.map_blocks(
        _concat_dsdatamatchup,
        ds,
        template=template,
    )
    return dsdatamatchup

"""


dl = 2
lon, lat = np.arange(-180, 180, dl), np.arange(-90, 90, dl)
acc_bins = np.arange(-1e-4, 1e-4, 1e-6)
dist_bins = np.arange(0, 1e5, 1e3)
