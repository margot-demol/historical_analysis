import numpy as np
import pandas as pd
import xarray as xr
import pyproj
from rasterio.transform import Affine
from math import ceil

import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.transforms as mtransforms

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
from histlib.cstes import labels, zarr_dir, c0, c1

"""
Diagnosis functions
--------------------------------------------------------------
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
    
"""
PDFS
-------------------------------
"""
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
                .compute()
            )

    elif isinstance(bins, np.ndarray):
        for var in list_var:
            ds_diag["pdf_" + var] = (
                histogram(ds[var].rename("acc"), bins=bins, density=True)
                .compute()
            )
    else:
        print("wrong bins type")

    ds_diag["nb_coloc"] = ds.sizes["obs"]
    if "id_comb" in ds:
        ds_diag["id_comb"] = ds["id_comb"]
    
    ds_diag.acc_bin.attrs={'longname':'accbins', 'units':r'$m.s^{-2}$'}

    return ds_diag

"""
MS
-------------------------------
"""
def ms_dataset(dsm, l) :
    dsm = match.add_except_sum(dsm)   
    nb = dsm.sizes["obs"]
    ds = (dsm**2).mean('obs')
    ds["nb_coloc"] = nb
    ds['drifter_sat_year']=l
    ds = ds.expand_dims('drifter_sat_year')
    ds = ds.set_coords('drifter_sat_year')
    if "id_comb" in dsm:
        ds["id_comb"] = dsm["id_comb"]
    ## ATTRS
    for v in dsm.keys() :
        ds[v].attrs =dsm[v].attrs
        ds[v].attrs['units']=r'$m^2.s^{-4}$'
        
    return ds
    
def global_cor(dsc):
    #global ms
    cor = (((dsc*dsc.nb_coloc).sum('drifter_sat_year'))/(dsc.nb_coloc.sum('drifter_sat_year'))).drop('nb_coloc')
    print(dsc.nb_coloc.sum('drifter_sat_year'))
    for v in dsc.keys():
        if v != 'nb_coloc':
            cor[v].attrs=dsc[v].attrs
    
    #nb_coloc
    nb_coloc = dsc.nb_coloc.sum('drifter_sat_year')
    cor['nb_coloc']=nb_coloc
    return cor
    
def global_ms_drifter_sat_year(dsmean, dsms, alpha):
    """
    dsmean: dataset with mean values
    dsms : dataset with ms values

    """
    
    #global ms
    ms = (((dsms*dsms.nb_coloc).sum('drifter_sat_year'))/(dsms.nb_coloc.sum('drifter_sat_year'))).drop('nb_coloc')
    print(dsms.nb_coloc.sum('drifter_sat_year'))
    for v in dsms.keys():
        if v != 'nb_coloc':
            ms[v].attrs=dsms[v].attrs
    #global mean and var
    mean = (((dsmean*dsmean.nb_coloc).sum('drifter_sat_year'))/(dsmean.nb_coloc.sum('drifter_sat_year'))).drop('nb_coloc')
    var = ms-mean**2

    
    #nb_coloc
    nb_coloc = dsms.nb_coloc.sum('drifter_sat_year')
    ms['nb_coloc']=nb_coloc

    #error
    from scipy.stats import ncx2
    def get_xlow(mean, var, nb_coloc, alpha):
        df = nb_coloc -1
        nc = df*(mean**2)/var
        norm = df/var
        return ncx2.ppf(alpha/2, df, nc)/norm

    def get_xup(mean, var, nb_coloc, alpha):
        df = nb_coloc -1
        nc = df*(mean**2)/var
        norm = df/var
        return ncx2.ppf(1-alpha/2, df, nc)/norm

    mslow = xr.apply_ufunc(get_xlow, mean, var, nb_coloc, alpha)
    msup = xr.apply_ufunc(get_xup, mean, var, nb_coloc, alpha)
    
    return ms, mslow, msup

def compute_sum_ms(ds, dserr, id_):
    dic = ds['sum_'+id_].attrs
    return ds[dic['acc']]+ds[dic['coriolis']]+ds[dic['ggrad']]+ds[dic['wind']], dserr[dic['acc']]+dserr[dic['coriolis']]+dserr[dic['ggrad']]+dserr[dic['wind']], 
    
def nMSRe_id(ds,dserr, id_):
    sum_ms, sum_ms_err = compute_sum_ms(ds,dserr, id_)
    ter = ds['sum_'+id_]/sum_ms*100
    tererr = ter*(dserr['sum_'+id_]/ds['sum_'+id_]+sum_ms_err/sum_ms)
    return ter.values, tererr.values

def C_x(ds,dserr, id_):
    def C_x_one_id(ds,dserr, id_):
        dic = ds['sum_'+id_].attrs
        lab = ['acc', 'coriolis', 'ggrad', 'wind']
        cx = {l:(ds['exc_'+l+'_'+id_] - ds['sum_'+id_]).values for l in lab}
        err_cx = {l+'_err':(dserr['exc_'+l+'_'+id_] + dserr['sum_'+id_]).values for l in lab}
        cx.update(err_cx)
        df = pd.DataFrame(cx, index=pd.Index([id_], name='id_comb'))
        return df

    if isinstance(id_, str):
        return C_x_one_id(ds,dserr, id_)
    else : 
        return pd.concat([C_x_one_id(ds,dserr, i) for i in id_])

def true_err_x(ds,dserr, id_) :
    dso = xr.Dataset()
    dic = ds['sum_'+id_].attrs
    for x in ['acc', 'coriolis', 'ggrad', 'wind']:
        X =dic[x]
        dso[x] = ds[X]
        dso[x+'_err'] = dserr[X]
        dso['exc_'+x] = ds['exc_' + x +'_'+ id_]
        dso['true_'+x] = (ds[X] - ds['sum_'+id_] + ds['exc_' + x +'_'+ id_])/2
        dso['true_'+x+'_err'] = (dserr[X] + dserr['sum_'+id_] + dserr['exc_' + x +'_'+ id_])/2
        dso['err_'+x] = (ds[X] + ds['sum_'+id_] - ds['exc_' + x +'_'+ id_])/2
        dso['err_'+x+'_err'] = (dserr[X] + dserr['sum_'+id_] + dserr['exc_' + x +'_'+ id_])/2
    dso['S'] = ds['sum_'+id_]
    return dso
        
"""
PLOT
-------------------------------
"""
"""
def plot_true_err_cor_part(ds, dscor, id_dic, ax, title=None, legend=True, rd=0):
    ""
    Parameters
    ----------
    ds : dataset with rms of x, excx and sum (created by )
    id_: identification of the combination
    ax : axis on which to plot

    ""
    l = '_nolegend_'
    dp = 9e-12
    
    def vn(id_dic, key1, key2):
        return 'prod_'+id_dic[key1]+'__'+id_dic[key2]
    w = 0.2
    # ACC X 
    #ax.bar(1.5, ds['acc'], color = 'k', width = 0.46, zorder=3, align = 'center')
    if legend : l ='total explained physical signal parts'
    ax.bar(1.5, ds['true_acc'], yerr = ds['true_acc_err'], capsize=10,
           color = 'k', width = 0.45, zorder=3, align = 'center', label=l)
    if legend : l ='Unexplained errors parts'
    ax.bar(1.5, ds['err_acc'], bottom = ds['true_acc'],yerr = ds['acc_err'], capsize=10,
           color = 'lightgrey', width = 0.45, zorder=3, align = 'center', label = l)
    ax.bar(1.5, -dscor[vn(id_dic, 'acc', 'wind')],
           color = c0['wind'], width = w, zorder=3, align = 'center')
    ax.bar(1.5, -dscor[vn(id_dic, 'acc', 'coriolis')],
           color = c0['coriolis'], width = w, zorder=3, align = 'center')
    ax.bar(1.5, -dscor[vn(id_dic, 'acc', 'ggrad')], bottom = -dscor[vn(id_dic, 'acc', 'coriolis')],
           color = c0['ggrad'], width = w, zorder=3, align = 'center')
    rse = np.round(ds['true_acc']/ds['acc']*100, rd).values
    if rd==0: rse=int(rse)
    ax.text(1.5, ds['true_acc']+dp, f'{rse}%', horizontalalignment='center')

    # CORIOLIS X  
    #ax.bar(2, ds['coriolis'], color = 'k', width = 0.46, zorder=3, align = 'center')
    
    ax.bar(2, ds['true_coriolis'], yerr = ds['true_coriolis_err'], capsize=10,
           color = 'k', width = 0.45, zorder=3, align = 'center')
    ax.bar(2, ds['err_coriolis'], bottom = ds['true_coriolis'],yerr = ds['coriolis_err'], capsize=10,
           color = 'lightgrey', width = 0.45, zorder=3, align = 'center')
    
    ax.bar(2, -dscor[vn(id_dic, 'coriolis', 'wind')],
           color = c0['wind'], width = w, zorder=3, align = 'center')
    if legend : l ='part explained by inertial acceleration'
    ax.bar(2, -dscor[vn(id_dic, 'acc', 'coriolis')], bottom = -dscor[vn(id_dic, 'coriolis', 'wind')],
           color = c0['acc'], width = w, zorder=3, align = 'center', label = l)
    if legend : l ='part explained by pressure gradient term'
    ax.bar(2, -dscor[vn(id_dic, 'coriolis', 'ggrad')], bottom = -dscor[vn(id_dic, 'acc', 'coriolis')]-dscor[vn(id_dic, 'coriolis', 'wind')],
           color = c0['ggrad'], width = w, zorder=3, align = 'center', label=l)
    rse = np.round(ds['true_coriolis']/ds['coriolis']*100, rd).values
    if rd==0: rse=int(rse)
    ax.text(2, ds['true_coriolis']+dp, f'{rse}%', horizontalalignment='center')

    # GGRAD X    
    #ax.bar(2.5, ds['ggrad'],color = 'k', width = 0.46, zorder=3, align = 'center')  
    
    ax.bar(2.5, ds['true_ggrad'],yerr = ds['ggrad_err'], capsize=10,
           color = 'k', width = 0.45, zorder=3, align = 'center')
    ax.bar(2.5, ds['err_ggrad'], bottom = ds['true_ggrad'],yerr = ds['ggrad_err'], capsize=10,
           color = 'lightgrey', width = 0.45, zorder=3, align = 'center')
    if legend : l ='part explained by coriolis acceleration'
    ax.bar(2.5, -dscor[vn(id_dic, 'coriolis', 'ggrad')], bottom = -dscor[vn(id_dic, 'acc', 'ggrad')],
           color = c0['coriolis'], width = w, zorder=3, align = 'center', label=l)
    if legend : l ='part explained by wind term'
    ax.bar(2.5, -dscor[vn(id_dic, 'ggrad', 'wind')],
           color = c0['wind'], width = w, zorder=3, align = 'center', label=l)
    ax.bar(2.5, -dscor[vn(id_dic, 'acc', 'ggrad')],
           color = c0['acc'], width = w, zorder=3, align = 'center')
    rse = np.round(ds['true_ggrad']/ds['ggrad']*100, rd).values
    if rd==0: rse=int(rse)
    ax.text(2.5, ds['true_ggrad']+dp, f'{rse}%', horizontalalignment='center')

    # WD X 
    #ax.bar(3, ds['wind'],color = 'k', width = 0.46, zorder=3, align = 'center')
    
    ax.bar(3, ds['true_wind'],yerr = ds['wind_err'], capsize=10,
           color = 'k', width = 0.45, zorder=3, align = 'center')
    ax.bar(3, ds['err_wind'], bottom = ds['true_wind'],yerr = ds['wind_err'], capsize=10,
           color = 'lightgrey', width = 0.45, zorder=3, align = 'center')
    
    ax.bar(3, -dscor[vn(id_dic, 'acc', 'wind')],
           color = c0['acc'], width = w, zorder=3, align = 'center')
    ax.bar(3, -dscor[vn(id_dic, 'coriolis', 'wind')],
           color = c0['coriolis'], width = w, zorder=3, align = 'center')
    ax.bar(3, -dscor[vn(id_dic, 'ggrad', 'wind')], bottom = -dscor[vn(id_dic, 'acc', 'wind')],
           color = c0['ggrad'], width = w, zorder=3, align = 'center')
    rse = np.round(ds['true_wind']/ds['wind']*100, rd).values
    if rd==0: rse=int(rse)
    ax.text(3, ds['true_wind']+dp, f'{rse}%', horizontalalignment='center')
    
    ax.grid(axis='y', zorder=0)
    if isinstance(title, int): ax.set_title(ds.id_comb)
    else : ax.set_title(title+'\n')

    if legend==False :
        ax.legend()
        ax.get_legend().remove()
        
    N=np.arange(1.5,3.5, 0.5) 
    ticks = (r'$d_tu$', r'$-fv$', r'$g \partial_x \eta$', r'$\frac{1}{\rho}\partial_z\tau_x$')
    ax.set_xticks(N, ticks,)
"""
def plot_true_err_cor_part(ds, dscor, id_dic, ax, title=None, legend=True, rd=0):
    """ 
    Parameters
    ----------
    ds : dataset with rms of x, excx and sum (created by )
    id_: identification of the combination
    ax : axis on which to plot

    """
    l = '_nolegend_'
    dp = 9e-12
    plt.rcParams['hatch.linewidth'] = 10
    def vn(id_dic, key1, key2):
        return 'prod_'+id_dic[key1]+'__'+id_dic[key2]
    w = 0.4
    # ACC X 
    #ax.bar(1.5, ds['acc'], color = 'k', width = 0.46, zorder=3, align = 'center')
    if legend : l ='Inertial acceleration'
    ax.bar(1.5, ds['true_acc'], yerr = ds['true_acc_err'], capsize=10,
           color = c0['acc'], width = 0.45, zorder=3, align = 'center', label=l)
    ax.bar(1.5, ds['err_acc'], bottom = ds['true_acc'],yerr = ds['acc_err'], capsize=10,
           color = 'lightgrey', width = 0.45, zorder=3, align = 'center')
    
    plt.rcParams['hatch.color'] = c0['acc']
    ax.bar(1.5, -dscor[vn(id_dic, 'acc', 'wind')],
           color = c0['wind'], width = w, zorder=3, align = 'center', hatch='/')
    ax.bar(1.5, -dscor[vn(id_dic, 'acc', 'coriolis')],
           color = c0['coriolis'], width = w, zorder=3, align = 'center', hatch='/')
    ax.bar(1.5, -dscor[vn(id_dic, 'acc', 'ggrad')], bottom = -dscor[vn(id_dic, 'acc', 'coriolis')],
           color = c0['ggrad'], width = w, zorder=3, align = 'center', hatch='/')
    
    rse = np.round(ds['true_acc']/ds['acc']*100, rd).values
    if rd==0: rse=int(rse)
    ax.text(1.5, ds['true_acc']+dp, f'{rse}%', horizontalalignment='center')

    # CORIOLIS X  
    #ax.bar(2, ds['coriolis'], color = 'k', width = 0.46, zorder=3, align = 'center')
    if legend : l ='Coriolis acceleration'
    ax.bar(2, ds['true_coriolis'], yerr = ds['true_coriolis_err'], capsize=10,
           color = c0['coriolis'], width = 0.45, zorder=3, align = 'center', label=l)
    ax.bar(2, ds['err_coriolis'], bottom = ds['true_coriolis'],yerr = ds['coriolis_err'], capsize=10,
           color = 'lightgrey', width = 0.45, zorder=3, align = 'center')
    
    
    plt.rcParams['hatch.color'] = c0['coriolis']    
    ax.bar(2, -dscor[vn(id_dic, 'coriolis', 'wind')],
           color = c0['wind'], width = w, zorder=3, align = 'center', hatch='/')
    ax.bar(2, -dscor[vn(id_dic, 'acc', 'coriolis')], bottom = -dscor[vn(id_dic, 'coriolis', 'wind')],
           color = c0['acc'], width = w, zorder=3, align = 'center',  hatch='/')
    ax.bar(2, -dscor[vn(id_dic, 'coriolis', 'ggrad')], bottom = -dscor[vn(id_dic, 'acc', 'coriolis')]-dscor[vn(id_dic, 'coriolis', 'wind')],
           color = c0['ggrad'], width = w, zorder=3, align = 'center', hatch='/')
    rse = np.round(ds['true_coriolis']/ds['coriolis']*100, rd).values
    if rd==0: rse=int(rse)
    ax.text(2, ds['true_coriolis']+dp, f'{rse}%', horizontalalignment='center')

    # GGRAD X    
    #ax.bar(2.5, ds['ggrad'],color = 'k', width = 0.46, zorder=3, align = 'center')  
    if legend : l ='Pressure gradient term'
    ax.bar(2.5, ds['true_ggrad'],yerr = ds['ggrad_err'], capsize=10,
           color = c0['ggrad'], width = 0.45, zorder=3, align = 'center', label=l)
    ax.bar(2.5, ds['err_ggrad'], bottom = ds['true_ggrad'],yerr = ds['ggrad_err'], capsize=10,
           color = 'lightgrey', width = 0.45, zorder=3, align = 'center')
    if legend : l ='part explained by coriolis acceleration'
    plt.rcParams['hatch.color'] = c0['ggrad']
    ax.bar(2.5, -dscor[vn(id_dic, 'coriolis', 'ggrad')], bottom = -dscor[vn(id_dic, 'acc', 'ggrad')],
           color = c0['coriolis'], width = w, zorder=3, align = 'center',  hatch='/')
    if legend : l ='part explained by wind term'
    ax.bar(2.5, -dscor[vn(id_dic, 'ggrad', 'wind')],
           color = c0['wind'], width = w, zorder=3, align = 'center', hatch='/')
    ax.bar(2.5, -dscor[vn(id_dic, 'acc', 'ggrad')],
           color = c0['acc'], width = w, zorder=3, align = 'center', hatch='/')
    rse = np.round(ds['true_ggrad']/ds['ggrad']*100, rd).values
    if rd==0: rse=int(rse)
    ax.text(2.5, ds['true_ggrad']+dp, f'{rse}%', horizontalalignment='center')

    # WD X 
    #ax.bar(3, ds['wind'],color = 'k', width = 0.46, zorder=3, align = 'center')
    if legend : l ='Turbulent stress vertical divergence'
    ax.bar(3, ds['true_wind'],yerr = ds['wind_err'], capsize=10,
           color = c0['wind'], width = 0.45, zorder=3, align = 'center', label=l)
    if legend : l ='Unexplained errors parts'
    ax.bar(3, ds['err_wind'], bottom = ds['true_wind'],yerr = ds['wind_err'], capsize=10,
           color = 'lightgrey', width = 0.45, zorder=3, align = 'center', label=l)
    plt.rcParams['hatch.color'] = c0['wind']
    ax.bar(3, -dscor[vn(id_dic, 'acc', 'wind')],
           color = c0['acc'], width = w, zorder=3, align = 'center', hatch='/')
    ax.bar(3, -dscor[vn(id_dic, 'coriolis', 'wind')],
           color = c0['coriolis'], width = w, zorder=3, align = 'center', hatch='/')
    ax.bar(3, -dscor[vn(id_dic, 'ggrad', 'wind')], bottom = -dscor[vn(id_dic, 'acc', 'wind')],
           color = c0['ggrad'], width = w, zorder=3, align = 'center', hatch='/')
    rse = np.round(ds['true_wind']/ds['wind']*100, rd).values
    if rd==0: rse=int(rse)
    ax.text(3, ds['true_wind']+dp, f'{rse}%', horizontalalignment='center')
    
    ax.grid(axis='y', zorder=0)
    if isinstance(title, int): ax.set_title(ds.id_comb)
    else : ax.set_title(title+'\n')

    if legend==False :
        ax.legend()
        ax.get_legend().remove()
        
    N=np.arange(1.5,3.5, 0.5) 
    ticks = (r'$d_tu$', r'$-fv$', r'$g \partial_x \eta$', r'$\frac{1}{\rho}\partial_z\tau_x$')
    ax.set_xticks(N, ticks,)

def synthetic_figure(ds, dsc, dic, ax) :
    plt.rcParams["axes.edgecolor"] = "w"
    a=1.5
    b = 2e-12
    bbox = dict(facecolor='w', alpha=0.8, edgecolor='w')
    def vn(id_dic, key1, key2):
        return 'prod_'+id_dic[key1]+'__'+id_dic[key2]
            
    ## INDIVIDUAL MS ##
    ax.barh(2*a, ds['acc'], color= c0['acc'], label = 'Inertial acceleration')
    ax.barh(2*a, ds['coriolis'], left =ds['acc']+b , color= c0['coriolis'], label = 'Coriolis acceleration')
    ax.barh(2*a, ds['ggrad'], left =ds['acc']+ds['coriolis']+2*b , color= c0['ggrad'], label = 'Pressure gradient term')
    ax.barh(2*a, ds['wind'], left =ds['acc']+ds['coriolis']+ds['ggrad']+3*b, color= c0['wind'], label = 'Vertical turbulent stress divergence')
    
    ts = ds['acc']+ds['coriolis']+ds['ggrad']+ds['wind']
    ax.text(ts/2, 2*a+0.5, r'Individual MS - Total initial MS $\langle \alpha^2 \rangle$', ha='center') 
    #percentage + MS
    key = ['acc', 'coriolis', 'ggrad', 'wind']
    for i in range(len(key)) :
        ax.text(sum([ds[v] for v in key[:i]])+ds[key[i]]/2+i*b, 2*a, f'{int(np.rint((ds[key[i]]/ts).values*100))}%', ha='center',bbox=bbox )
        ax.text(sum([ds[v] for v in key[:i]])+ds[key[i]]/2+i*b, 2*a-0.55, f'{np.format_float_scientific(ds[key[i]].values,precision = 1)}', ha='center')
    
    
    ## CAPTURED PHYSICAL + ERRORS PARTS ##
    ax.barh(1*a, ds['true_acc'], color= c0['acc'])
    ax.barh(1*a, ds['err_acc'], left = ds['true_acc'], color= 'lightgrey', label='Errors')
    ax.barh(1*a, ds['true_coriolis'], left =ds['true_acc']+ds['err_acc']+b, color= c0['coriolis'])
    ax.barh(1*a, ds['err_coriolis'], left =ds['true_acc']+ds['err_acc']+ ds['true_coriolis'], color= 'lightgrey')
    ax.barh(1*a, ds['true_ggrad'], left =ds['true_acc']+ds['err_acc']+ ds['true_coriolis']+ ds['err_coriolis']+b, color= c0['ggrad'])
    ax.barh(1*a, ds['err_ggrad'], left =ds['true_acc']+ds['err_acc']+ ds['true_coriolis']+ ds['err_coriolis']+ds['true_ggrad'], color= 'lightgrey')
    ax.barh(1*a, ds['true_wind'], left =ds['true_acc']+ds['err_acc']+ ds['true_coriolis']+ ds['err_coriolis']+ds['true_ggrad']+ds['err_ggrad']+b, color= c0['wind'])
    ax.barh(1*a, ds['err_wind'], left =ds['true_acc']+ds['err_acc']+ ds['true_coriolis']+ ds['err_coriolis']+ds['true_ggrad']+ds['err_ggrad']+ds['true_wind'], color= 'lightgrey')

    
    ax.text(ts/2, 1*a+0.5, r'Captured physical and errors parts MS $\langle \hat{\alpha}^2 \rangle$ and $\langle \alpha_e^2 \rangle$', ha='center') 
    #percentage + MS
    key = ['true_acc','err_acc', 'true_coriolis','err_coriolis', 'true_ggrad','err_ggrad', 'true_wind', 'err_wind']
    for i in range(len(key)) :
        if i==len(key)-1 :
            ax.text(sum([ds[v] for v in key[:i]])+ds[key[i]]/2+i*b/2, (1-0.1)*a, f'{int(np.rint((ds[key[i]]/ts).values*100))}%', ha='center', bbox=bbox)
        else : 
            ax.text(sum([ds[v] for v in key[:i]])+ds[key[i]]/2+i*b/2, 1*a, f'{int(np.rint((ds[key[i]]/ts).values*100))}%', ha='center', bbox=bbox)
        if i%2 ==1 :
            ax.text(sum([ds[v] for v in key[:i]])+ds[key[i]]/2+i*b/2, (1-0.1)*a-0.55, f'{np.format_float_scientific(ds[key[i]].values,precision = 1)}', ha='center')
        else :
            ax.text(sum([ds[v] for v in key[:i]])+ds[key[i]]/2+i*b/2, 1*a-0.55, f'{np.format_float_scientific(ds[key[i]].values,precision = 1)}', ha='center')
        
    ## PAIRS + RESIDUAL ##
    plt.rcParams['hatch.linewidth'] = 10
    plt.rcParams['hatch.color'] = c0['ggrad']
    ax.barh(0, dsc[vn(dic, 'coriolis', 'ggrad')], color=c0['coriolis'], hatch='/')
    plt.rcParams['hatch.color'] = c0['coriolis']
    ax.barh(0, dsc[vn(dic, 'acc', 'coriolis')], color=c0['acc'], hatch='/', left = dsc[vn(dic, 'coriolis', 'ggrad')]+b)
    plt.rcParams['hatch.color'] = c0['acc']
    ax.barh(0, dsc[vn(dic, 'acc', 'ggrad')], color=c0['ggrad'], hatch='/', left = dsc[vn(dic, 'coriolis', 'ggrad')]+dsc[vn(dic, 'acc', 'coriolis')]+2*b)
    plt.rcParams['hatch.color'] = c0['wind']
    ax.barh(0, dsc[vn(dic, 'coriolis', 'wind')], color=c0['coriolis'], hatch='/', left = dsc[vn(dic, 'coriolis', 'ggrad')]+dsc[vn(dic, 'acc', 'coriolis')]+dsc[vn(dic, 'acc', 'ggrad')]+3*b)
    ax.barh(0, ds['S'], color='lightgrey', left = dsc[vn(dic, 'coriolis', 'ggrad')]+dsc[vn(dic, 'acc', 'coriolis')]+dsc[vn(dic, 'acc', 'ggrad')]+dsc[vn(dic, 'coriolis', 'wind')]+4*b)

    tts = dsc[vn(dic, 'coriolis', 'ggrad')]+dsc[vn(dic, 'acc', 'coriolis')]+dsc[vn(dic, 'acc', 'ggrad')]+dsc[vn(dic, 'coriolis', 'wind')]+4*b+ds['S']

    sum_pairs = dsc[vn(dic, 'coriolis', 'ggrad')]+dsc[vn(dic, 'acc', 'coriolis')]+dsc[vn(dic, 'acc', 'ggrad')]+dsc[vn(dic, 'coriolis', 'wind')]+3*b
    ax.text(sum_pairs/2, 0.6, r"Pairs' contributions $C_{ij}$", ha='center')
    #accolade
    c = 1e-12
    id1 =0
    id2 = sum_pairs
    bx = [id1, id1, id2, id2]
    by = [0.45, 0.5, 0.5, 0.45]
    ax.plot(bx, by, 'k-', lw=2)
    ax.text(sum_pairs + ds['S']/2, 0.5, r'MSRe $\langle S^2 \rangle$', ha='center')

    #percentage + MS
    key = [vn(dic, 'coriolis', 'ggrad'),vn(dic, 'acc', 'coriolis'), vn(dic, 'acc', 'ggrad'), vn(dic, 'coriolis', 'wind')]
    for i in range(len(key)) :
        ax.text(sum([dsc[v] for v in key[:i]])+dsc[key[i]]/2+i*b, 0, f'{int(np.rint((dsc[key[i]]/ts).values*100))}%', ha='center', bbox=bbox)
        ax.text(sum([dsc[v] for v in key[:i]])+dsc[key[i]]/2+i*b, 0-0.55, f'{np.format_float_scientific(dsc[key[i]].values,precision = 1)}', ha='center')
    
    ax.text(sum([dsc[v] for v in key])+ds['S']/2+i*b, 0, f'{int(np.rint((ds["S"]/ts).values*100))}%', ha='center', bbox=bbox)
    ax.text(sum([dsc[v] for v in key])+ds['S']/2+i*b, 0-0.55, f'{np.format_float_scientific(ds["S"].values,precision = 1)}', ha='center')

    # FIGURE SET
    ax.set_yticks([])
    ax.set_xlim(-1e-11, tts+1e-11)
    ax.set_ylim(-1, 4)
    ax.get_yaxis().set_visible(False)
    ax.axhline(-1, color='k')

    plt.rcParams["axes.edgecolor"] = "k"
    

def plot_cor_uncor_part(ds, ax, title=None):
    """ 
    Parameters
    ----------
    ds : dataset with rms of x, excx and sum (created by )
    id_: identification of the combination
    ax : axis on which to plot

    """
    # ACC X
    ax.bar(1.5, ds['true_acc'], yerr = ds['true_acc_err'], capsize=10,
           color = c0['acc'], width = 0.4, zorder=3, align = 'center')
    ax.bar(1.5, ds['err_acc'], bottom = ds['true_acc'],yerr = ds['acc_err'], capsize=10,
           color = 'lightgrey', width = 0.4, zorder=3, align = 'center')
    ax.text(1.5, ds['acc']+5e-12, str(np.format_float_scientific(ds['acc'].values,precision = 3)), horizontalalignment='center')
    rse = np.round(ds['true_acc']/ds['acc']*100,2)
    ax.text(1.5, ds['true_acc']+5e-12, f'{rse.values}%', horizontalalignment='center')

    # CORIOLIS
    ax.bar(2, ds['true_coriolis'], yerr = ds['true_coriolis_err'], capsize=10,
           color = c0['coriolis'], width = 0.4, zorder=3, align = 'center')
    ax.bar(2, ds['err_coriolis'], bottom = ds['true_coriolis'],yerr = ds['coriolis_err'], capsize=10,
           color = 'lightgrey', width = 0.4, zorder=3, align = 'center')
    ax.text(2, ds['coriolis']+5e-12, str(np.format_float_scientific(ds['coriolis'].values,precision = 3)), horizontalalignment='center')
    rse = np.round(ds['true_coriolis']/ds['coriolis']*100,2)
    ax.text(2, ds['true_coriolis']+5e-12, f'{rse.values}%', horizontalalignment='center')

    # G GRADIENT SLA
    ax.bar(2.5, ds['true_ggrad'],yerr = ds['ggrad_err'], capsize=10,
           color = c0['ggrad'], width = 0.4, zorder=3, align = 'center')
    ax.bar(2.5, ds['err_ggrad'], bottom = ds['true_ggrad'],yerr = ds['ggrad_err'], capsize=10,
           color = 'lightgrey', width = 0.4, zorder=3, align = 'center')
    ax.text(2.5, ds['ggrad']+5e-12, str(np.format_float_scientific(ds['ggrad'].values,precision = 3)), horizontalalignment='center')
    rse = np.round(ds['true_ggrad']/ds['ggrad']*100,2)
    ax.text(2.5, ds['true_ggrad']+5e-12, f'{rse.values}%', horizontalalignment='center')

        # WIND
    ax.bar(3, ds['true_wind'],yerr = ds['wind_err'], capsize=10,
           color = c0['wind'], width = 0.4, zorder=3, align = 'center')
    ax.bar(3, ds['err_wind'], bottom = ds['true_wind'],yerr = ds['wind_err'], capsize=10,
           color = 'lightgrey', width = 0.4, zorder=3, align = 'center')
    ax.text(3, ds['wind']+5e-12, str(np.format_float_scientific(ds['wind'].values,precision = 3)), horizontalalignment='center')
    rse = np.round(ds['true_wind']/ds['wind']*100,2)
    ax.text(3, ds['true_wind']+5e-12, f'{rse.values}%', horizontalalignment='center')

    #ax.set_ylim((0,8e-5))
    ax.grid(axis='y', zorder=0)

    if isinstance(title, int): ax.set_title(ds.id_comb)
    else : ax.set_title(title+'\n')
    
    N=np.arange(1.5,3.5, 0.5) 
    ticks = (r'$d_tu$', r'$-fv$', r'$g \partial_x \eta$', r'$\frac{1}{\rho}\partial_z\tau_x$')
    ax.set_xticks(N, ticks,)


def plot_closure_bar(ds, id_, ax, title=1):
    """ Plot closure bars for a combination on the axis ax.
    
    Parameters
    ----------
    ds : dataset with rms of x, excx and sum (created by )
    id_: identification of the combination
    ax : axis on which to plot

    """
    dic = ds['sum_'+id_].attrs
    acc = ''+dic['acc']
    cor = ''+dic['coriolis']
    ggrad = ''+dic['ggrad']
    wd = ''+dic['wind']

    # ACC X
    ax.bar(1.5, (np.sqrt(ds[acc]) + np.sqrt(ds['exc_acc_' + id_]))**2,
           color ='lightgrey', width = 0.45, zorder=3, align = 'center',)
    ax.bar(1.5, (np.sqrt(ds[acc]) - np.sqrt(ds['exc_acc_' + id_]))**2,
           color ='w', width = 0.45, zorder=3, align = 'center',)
    ax.bar(1.5, ds['exc_acc_' + id_],
           color = 'lightsteelblue', width = 0.4, zorder=3, align = 'center')
    ax.bar(1.5, ds[acc], bottom = ds['exc_acc_' + id_],
           color = 'red', width = 0.4, zorder=3, align = 'center')

    # CORIOLIS
    ax.bar(2, (np.sqrt(ds['exc_coriolis_' + id_]) + np.sqrt(ds[cor]))**2,
           color='lightgrey', width = 0.45, zorder=3, align = 'center',)
    ax.bar(2, (np.sqrt(ds['exc_coriolis_' + id_]) - np.sqrt(ds[cor]))**2,
           color='w', width = 0.45, zorder=3, align = 'center', )
    ax.bar(2, ds['exc_coriolis_' + id_],
           color = 'lightsteelblue', width = 0.4, zorder=3, align = 'center')
    ax.bar(2, ds[cor], bottom = ds['exc_coriolis_' + id_],
           color = 'green', width = 0.4, zorder=3, align = 'center')

    # G GRADIENT SLA
    ax.bar(2.5, (np.sqrt(ds['exc_ggrad_' + id_]) + np.sqrt(ds[ggrad]))**2,
           color ='lightgrey', width = 0.45, zorder=3, align = 'center',)
    ax.bar(2.5, (np.sqrt(ds['exc_ggrad_' + id_]) - np.sqrt(ds[ggrad]))**2,
           color ='w', width = 0.45, zorder=3, align = 'center',)
    ax.bar(2.5, ds['exc_ggrad_' + id_],
           color = 'lightsteelblue', width = 0.4, zorder=3, align = 'center')
    ax.bar(2.5, ds[ggrad], bottom = ds['exc_ggrad_' + id_],
           color = 'c', width = 0.4, zorder=3, align = 'center')

    # WIND
    ax.bar(3, (np.sqrt(ds['exc_wind_' + id_]) + np.sqrt(ds[wd]))**2,
           color ='lightgrey', width = 0.45, zorder=3, align = 'center',)
    ax.bar(3, (np.sqrt(ds['exc_wind_' + id_]) - np.sqrt(ds[wd]))**2,
           color ='w', width = 0.45, zorder=3, align = 'center',)
    ax.bar(3, ds['exc_wind_' + id_],
           color = 'lightsteelblue', width = 0.4, zorder=3, align = 'center')
    ax.bar(3, ds[wd], bottom = ds['exc_wind_' + id_],
           color = 'mediumvioletred', width = 0.4, zorder=3, align = 'center')
    
    #SUM of ms^2
    ax.bar(1,ds[acc],
           color = 'red', width = 0.4, zorder=3, align = 'center')
    ax.bar(1, ds[cor], bottom = ds[acc],
           color = 'green', width = 0.4, zorder=3, align = 'center')
    ax.bar(1, ds[ggrad], bottom = ds[acc]+ds[cor],
           color = 'c', width = 0.4, zorder=3, align = 'center')
    ax.bar(1, ds[wd], bottom = ds[acc]+ds[cor]+ds[ggrad],
           color = 'mediumvioletred', width = 0.4, zorder=3, align = 'center')
    Ss=ds[acc]+ds[cor]+ds[ggrad]+ds[wd]
    ax.text(0.75, Ss+Ss/30, str(np.format_float_scientific(Ss.values,precision = 3)))

    ax.set_ylabel(r'Mean square $\langle ...^2 \rangle$ [$m^2/s^4$]')
    if isinstance(title, int): ax.set_title(id_)
    else : ax.set_title(title+'\n')
    
    #TOTAL
    S = ds['sum_'+id_]
    ax.bar(0.5, S, color ='k',width = 0.4, zorder=3)
    ax.text(0.25, S+S/20, str(np.format_float_scientific(S.values,precision = 3)))
    ax.axhline(y=S, c="k", linewidth=2, ls=':', zorder=4)
    
    #ax.set_ylim((0,8e-5))
    ax.grid(axis='y', zorder=0)
    
    N=np.arange(0.5,3.5, 0.5) 
    #g = ds[ggrad].attrs['long_name'].replace('rms[',r'').replace(']','').replace('altimatchup','aviso').replace('driftermatchup','aviso').split('+')
    #ggrad_tick='$\n $+'.join(g)
    #w = ds_all['rms_es_cstrio_z15_alti_wd_x'].attrs['long_name'].replace('rms[',r'').replace(']','').split(' from')
    #wd_tick = '\n from'.join(w)
    #ticks = (r'$\langle S^2\rangle$',r'$\sum_x \langle x^2\rangle$',r'$d_tu$', r'$-fv$', ggrad_tick, r'$\frac{1}{\rho}\partial_z\tau_x$')
    #ax.set_xticks(N, ticks,)

    print(f'acc:{ds[acc].values}, coriolis:{ds[cor].values}, ggrad:{ds[ggrad].values}, wind:{ds[wd].values}')

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
    ticks = [r"$d_tu$", r"$-fv$", ggrad_tick, "$fv_{Ekman}$"]

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
        r"$\frac{\langle S_{-x}^2\rangle}{\langle S^2\rangle}$   $\alpha =$" + t
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

def put_fig_letter(ax, letter):
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, letter+')', transform=ax.transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

"""
Default list
-------------------------------
"""


dl = 2
lon, lat = np.arange(-180, 180, dl), np.arange(-90, 90, dl)
acc_bins = np.arange(-1e-4, 1e-4, 1e-6)
dist_bins = np.arange(0, 1e5, 1e3)
