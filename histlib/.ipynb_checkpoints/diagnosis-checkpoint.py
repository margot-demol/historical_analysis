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
    
def globa_ms_drifter_sat_year(dsms):
    return (dsms*dsms.nb_coloc).mean('drifter_sat_year').drop('nb_coloc')


"""
PLOT
-------------------------------
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
Default list
-------------------------------
"""


dl = 2
lon, lat = np.arange(-180, 180, dl), np.arange(-90, 90, dl)
acc_bins = np.arange(-1e-4, 1e-4, 1e-6)
dist_bins = np.arange(0, 1e5, 1e3)
