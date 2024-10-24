import numpy as np
import pandas as pd
import xarray as xr
import dask.dataframe as dd
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
zarr_dir = zarr_dir+'_ok'
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
      

def put_fig_letter(fig, ax, letter):
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, letter+')', transform=ax.transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0), zorder=30)

"""
BUILD MATCHUP
--------------------------------------------------------------
"""

def path_csv(comb,drifter_type, drogue_status, DL, DT):
    # Drogue status
    if drogue_status == 'both' : dr_st = 'bothdroguedundrogued'
    elif drogue_status : dr_st = 'drogued'
    elif not drogue_status : dr_st = 'undrogued'

    #Drifter type
    if drifter_type == 'both' : dr = 'bothgpsargos'
    elif drifter_type == 'gps' : dr = 'gps'
    elif drifter_type == 'argos' : dr = 'argos'

    #dl 
    if DL : dl=str(int(DL))
    if DL ==None : dl='None'
    if DT : dt=str(int(DT/60))
    if DT ==None : dt='None'

    return os.path.join(zarr_dir, 'analysis_files', f'{dr_st}__{dr}__{dl}km__{dt}min__'+'__'.join([comb[key] for key in comb])+'.csv')

def build_diagnostic(df, comb):
    if (isinstance(df, pd.core.frame.DataFrame) or  isinstance(df, dd.core.DataFrame)):
        df = df.rename(columns = {comb[key]:key for key in comb}).reset_index()[['obs','acc','cor','drogue_status','drifter_type','ggx','wd','alti___distance','alti___time_difference','lat','lon','time']]
        
        vars = ['obs','acc','cor','drogue_status','drifter_type','ggx','wd','alti___distance','alti___time_difference','lat','lon','time']
        types = ['int64','float64','float64','bool','U','float64','float64','float64','float64','float64','float64','datetime64']
    
        df.astype({vars[i] : types[i] for i in range(len(vars))})
        
        tms = [tm for tm in comb]
        
        # residual
        df['s'] = df[[tm for tm in comb]].sum(axis=1)
        
        # correlations
        from itertools import combinations
        correlation = list(combinations(tms, 2))
        for c in correlation :
            df[f'{c[0]}_{c[1]}'] = df[c[0]]*df[c[1]]
    
        for tm in tms :
            #Errors parts
            df['E_'+tm] = df.s*df[tm]
            # Observed balanced parts
            df['B_'+tm] = -df[[f'{c[0]}_{c[1]}' for c in correlation if tm in f'{c[0]}_{c[1]}']].sum(axis=1)
    
        #Contributions
        for c in correlation :
            df['X_'+f'{c[0]}_{c[1]}'] = -2*df[f'{c[0]}_{c[1]}']
    
        for tm in tms + ['s'] :
            df[tm.upper()] = df[tm]**2
        
        df['sigma'] = df[[tm.upper() for tm in tms]].sum(axis=1)
        
    if isinstance(df, xr.core.dataset.Dataset) : 
        df = df.rename({comb[key]:key for key in comb})[['obs','acc','cor','drogue_status','drifter_type','ggx','wd','alti___distance','alti___time_difference','lat','lon','time']]
        
        tms = [tm for tm in comb]
        
        # residual
        df['s'] = sum([df[v] for v in [tm for tm in comb]])
        
        # correlations
        from itertools import combinations
        correlation = list(combinations(tms, 2))
        for c in correlation :
            df[f'{c[0]}_{c[1]}'] = df[c[0]]*df[c[1]]
    
        for tm in tms :
            #Errors parts
            df['E_'+tm] = df.s*df[tm]
            # Observed balanced parts
            df['B_'+tm] = sum([-df[v] for v in [f'{c[0]}_{c[1]}' for c in correlation if tm in f'{c[0]}_{c[1]}']])
    
        #Contributions
        for c in correlation :
            df['X_'+f'{c[0]}_{c[1]}'] = -2*df[f'{c[0]}_{c[1]}']
    
        for tm in tms + ['s'] :
            df[tm.upper()] = df[tm]**2
        df['sigma'] = sum([df[v] for v in [tm.upper() for tm in tms]])

    return df
    
def build_matchup_dataframe(comb, drifter_type = 'both', drogue_status='both', DL=None, DT=None, store=False):
    from histlib.cstes import labels
    # gps/argos
    if drifter_type!='both': 
        labels = [l for l in labels if drifter_type in l]
    DF =[]
    for l in labels :
        paths = [f'/home/datawork-lops-osi/aponte/margot/historical_coloc_ok/matchup/matchup_{l}.zarr'] + glob(os.path.join(f'/home/datawork-lops-osi/aponte/margot/historical_coloc_ok/cutoff_matchup/cutoff_matchup_{l}*.zarr'))
        df = xr.merge([xr.open_dataset(path) for path in paths])[[comb[key] for key in comb]+['drogue_status']].to_dataframe()
        df['drifter_type'] = l.split('_')[0]
        df['drogue_status'] = df['drogue_status'].astype('bool')
        
        if drogue_status != 'both':
            df = df.where(df.drogue_status==drogue_status).dropna()
        if DL :
            df =df.where(df.alti___distance<DL).dropna()
        if DT : 
            df =df.where(df.alti___time_difference<DT).dropna()
        DF.append(df)
        print(l)
    
    df = pd.concat(DF)

    df = build_diagnostic(df, comb)
    df = df.set_index('obs')
    
    if store :
        df.to_csv(path_csv(comb,drifter_type, drogue_status, DL, DT))
    return df


"""
ERROR COMPUTING
--------------------------------------------------------------
"""

def mean_df(df):
    return df.mean()
from scipy.stats import bootstrap
def compute_bootstrap_error(dff, bootkwargs):
    #print(len(dff))
    if len(dff)<5:
        return np.nan
    else : 
        data = (dff, )  # samples must be in a sequence
        return bootstrap(data, statistic = mean_df, **bootkwargs).standard_error

"""
GLOBAL ANALYSIS
--------------------------------------------------------------
"""
def synthetic_figure(df, ax, xlim=None, aviso=False) :
    from histlib.cstes import U
    plt.rcParams["axes.edgecolor"] = "w"
    a=1.5
    bbox = dict(facecolor='w', alpha=0.8, edgecolor='w')
    
    ts = df['sigma']
    print(ts)
    #gap between bars for readability
    if xlim : b=xlim/400
    else : b=ts/400

    #b = 1e-10
    
    ## INDIVIDUAL MS ##
    ax.barh(2*a, df['ACC'], color= c0['acc'], label = 'Lagrangian acceleration')
    ax.barh(2*a, df['COR'], left =df['ACC']+b , color= c0['cor'], label = 'Coriolis acceleration')
    ax.barh(2*a, df['GGX'], left =df['ACC']+df['COR']+2*b , color= c0['ggx'], label = 'Pressure gradient term')
    ax.barh(2*a, df['WD'], left =df['ACC']+df['COR']+df['GGX']+3*b, color= c0['wd'], label = 'Wind term')
    

    ax.text(ts/2, 2*a+0.5, r'Individual MS $A_i$', ha='center') 
    #percentage + MS
    key = ['ACC', 'COR', 'GGX', 'WD']
    for i in range(len(key)) :
        ax.text(sum([df[v] for v in key[:i]])+df[key[i]]/2+i*b, 2*a, f'{int(np.rint((df[key[i]]/ts)*100))} %', ha='center',bbox=bbox )
        ax.text(sum([df[v] for v in key[:i]])+df[key[i]]/2+i*b, 2*a-0.55, f'{np.round(df[key[i]],2)}', ha='center')

    #accolade
    c = 1e-12*U**2
    id1 =0
    id2 = ts + 3*b
    bx = [id1, id1, id2, id2]
    by = [3.70, 3.75, 3.75, 3.70]
    ax.plot(bx, by, 'k-', lw=2)
    ax.text(ts, 3.8, r'$\Sigma$', fontsize=15, ha='center')
    
    ## CAPTURED PHYSICAL + ERRORS PARTS ##
    ax.barh(1*a, df['B_acc'], color= c0['acc'])
    ax.barh(1*a, df['E_acc'], left = df['B_acc'], color= 'lightgrey', label='Errors')
    ax.barh(1*a, df['B_cor'], left =df['B_acc']+df['E_acc']+b, color= c0['cor'])
    ax.barh(1*a, df['E_cor'], left =df['B_acc']+df['E_acc']+ df['B_cor'], color= 'lightgrey')
    ax.barh(1*a, df['B_ggx'], left =df['B_acc']+df['E_acc']+ df['B_cor']+ df['E_cor']+b, color= c0['ggx'])
    if df['E_ggx']>0 : 
        ax.barh(1*a, df['E_ggx'], left =df['B_acc']+df['E_acc']+ df['B_cor']+ df['E_cor']+df['B_ggx'], color= 'lightgrey')
        ax.barh(1*a, df['B_wd'], left =df['B_acc']+df['E_acc']+ df['B_cor']+ df['E_cor']+df['B_ggx']+df['E_ggx']+b, color= c0['wd'])
        ax.barh(1*a, df['E_wd'], left =df['B_acc']+df['E_acc']+ df['B_cor']+ df['E_cor']+df['B_ggx']+df['E_ggx']+df['B_wd'], color= 'lightgrey')
    else : 
        print('ok')
        ax.barh(1*a, df['B_wd'], left =df['B_acc']+df['E_acc']+ df['B_cor']+ df['E_cor']+df['B_ggx']+2*b, color= c0['wd'])
        ax.barh(1*a, df['E_wd'], left =df['B_acc']+df['E_acc']+ df['B_cor']+ df['E_cor']+df['B_ggx']+df['B_wd']+2*b, color= 'lightgrey')
    
    ax.text(ts/2, 1*a+0.5, r'Balanced physical and errors parts MS $B_i$ and $E_i$', ha='center') 
            
    #percentage + MS
    key = ['B_acc','E_acc', 'B_cor','E_cor', 'B_ggx','E_ggx', 'B_wd', 'E_wd']
    for i in range(len(key)) :
        d=0#vertical +
        dx=0#horizontal + on MS
        dxx=0#horizontal + on percentage
        if i==len(key)-1 : 
            d=-0.1*a 
            dx = 3e-11
            dxx = 1.5e-11
        if i==len(key)-2 : d=0.1*a 
        if aviso and i==len(key)-3 : d=-0.1*a
        ax.text(sum([df[v] for v in key[:i]])+df[key[i]]/2+i*b/2+dxx, a+d, f'{int(np.rint((df[key[i]]/ts)*100))} %', ha='center', bbox=bbox)
        d=0        
        if i%2 ==1 : d=-0.1*a
        ax.text(sum([df[v] for v in key[:i]])+df[key[i]]/2+i*b/2+dx, a+d -0.55, f'{np.round(df[key[i]],2)}', ha='center')
        
    ## PAIRS + RESIDUAL ##
    plt.rcParams['hatch.linewidth'] = 10
    plt.rcParams['hatch.color'] = c0['ggx']
    ax.barh(0, df['X_cor_ggx'], color=c0['cor'], hatch='/')
    plt.rcParams['hatch.color'] = c0['cor']
    ax.barh(0, df['X_acc_cor'], color=c0['acc'], hatch='/', left = df['X_cor_ggx']+b)
    plt.rcParams['hatch.color'] = c0['acc']
    ax.barh(0, df['X_acc_ggx'], color=c0['ggx'], hatch='/', left = df['X_cor_ggx']+df['X_acc_cor']+2*b)
    plt.rcParams['hatch.color'] = c0['wd']
    ax.barh(0, df['X_cor_wd'], color=c0['cor'], hatch='/', left = df['X_cor_ggx']+df['X_acc_cor']+df['X_acc_ggx']+3*b)
    ax.barh(0, df['S'], color='lightgrey', left = df['X_cor_ggx']+df['X_acc_cor']+df['X_acc_ggx']+df['X_cor_wd']+4*b)

    tts = df['X_cor_ggx']+df['X_acc_cor']+df['X_acc_ggx']+df['X_cor_wd']+4*b+df['S']
    print(tts)
    sum_pairs = df['X_cor_ggx']+df['X_acc_cor']+df['X_acc_ggx']+df['X_cor_wd']+3*b
    ax.text(sum_pairs/2, 0.6, r"Pairs' contributions $X_{ij}$", ha='center')
    #accolade
    c = 1e-12
    id1 = 0
    id2 = sum_pairs
    bx = [id1, id1, id2, id2]
    by = [0.45, 0.5, 0.5, 0.45]
    #ax.plot(bx, by, 'k-', lw=2)
    ax.text(sum_pairs + df['S']/2, 0.5, r'$S$', ha='center')

    #percentage + MS
    from itertools import combinations
    correlation = list(combinations(['acc', 'cor', 'ggx', 'wd'], 2))
    key = ['X_cor_ggx', 'X_acc_cor','X_acc_ggx','X_cor_wd']

    for i in range(len(key)) :
        ax.text(sum([df[v] for v in key[:i]])+df[key[i]]/2+i*b, 0, f'{int(np.rint((df[key[i]]/ts)*100))} %', ha='center', bbox=bbox)
        d=0
        if aviso and key[i]== 'X_acc_ggx' : d = -0.1*a
        ax.text(sum([df[v] for v in key[:i]])+df[key[i]]/2+i*b, 0-0.55+d, f'{np.round(df[key[i]],2)}', ha='center')
    
    ax.text(sum([df[v] for v in key])+df['S']/2+i*b, 0, f'{int(np.rint((df["S"]/ts)*100))} %', ha='center', bbox=bbox)
    ax.text(sum([df[v] for v in key])+df['S']/2+i*b, 0-0.55, f'{np.round(df["S"],2)}', ha='center')

    # FIGURE SET
    ax.set_yticks([])
    if not xlim : xlim=tts
    ax.set_xlim(-1e-11, xlim+1e-11)
    ax.set_ylim(-1, 4.1)
    ax.get_yaxis().set_visible(False)
    ax.annotate('',xy=(xlim,-1),xytext=(0,-1),arrowprops={'arrowstyle':'->', 'facecolor':'k'})
    ax.set_xlabel(r'$[U^2]$')


"""
LONLAT ANALYSIS
--------------------------------------------------------------
"""
def lonlat_stats(df, dl=5, vars_errors=['S'], bootkwargs=dict()):
    df["latbin"] = (df.lat // dl) * dl
    df["lonbin"] = (df.lon // dl) * dl
    
    # Mean in bins
    mean = df.groupby(["latbin", "lonbin"]).mean().compute()
    
    #count
    count = df.reset_index()[['obs', 'latbin', 'lonbin']].groupby(["latbin", "lonbin"]).count().obs.compute().rename('nb_coloc_bin')

    # bootstrap errors
    DF = []
    for v in vars_errors :
        DF.append(df.reset_index()[[v, 'latbin', 'lonbin']].groupby(["latbin", "lonbin"])[v].apply(compute_bootstrap_error, bootkwargs).compute())
    booterrors = pd.concat(DF, axis=1)
    booterrors = booterrors.rename(columns={v:'be__'+v for v in booterrors.columns})

    return pd.concat([mean, count,booterrors], axis=1)


"""
Default list
-------------------------------
"""


dl = 2
lon, lat = np.arange(-180, 180, dl), np.arange(-90, 90, dl)
acc_bins = np.arange(-1e-4, 1e-4, 1e-6)
dist_bins = np.arange(0, 1e5, 1e3)
