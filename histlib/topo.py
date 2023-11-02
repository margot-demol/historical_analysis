import os

import xarray as xr
import pandas as pd
import numpy as np


etopo_dir = "/Users/aponte/Data/bathy/etopo1"
geoid_dir = "/Users/aponte/Data/geoid"
mdt_path = "/Users/aponte/Data/mdt/cmems_obs-sl_med_phy-mdt_my_l4-0.0417deg.nc"

# --------------------------- topo related --------------------------------


def load_etopo1(f="ETOPO1_Ice_c_gmt4.zarr"):
    ds = xr.open_zarr(os.path.join(etopo_dir, "zarr/" + f))
    ds = ds.rename(dict(z="h"))
    return ds


# --------------------------- geoid related --------------------------------


def read_gdf(file):
    """read geographic data file"""

    f = open(file, "r")

    # read header
    attrs = {}
    header = True
    while header:
        line = f.readline()
        header = "end_of_head" not in line
        sline = line.split()
        if len(sline) > 1 and header:
            attrs[sline[0]] = sline[1]

    # read core data
    df = pd.read_table(
        f,
        names=["lon", "lat", "geoid"],
        delim_whitespace=True,
    ).set_index(["lon", "lat"])
    ds = df.to_xarray()

    ds.attrs.update(**attrs)

    return ds


def load_geoid(label="EIGEN-6C4_1"):
    """load geoid given it's label"""
    ds = xr.open_zarr(os.path.join(geoid_dir, f"zarr/{label}.zarr"))
    return ds


# --------------------------- mdt, mean sea level ------------------------------


def load_mdt(**kwargs):
    ds = xr.open_dataset(mdt_path).squeeze().rename(longitude="lon", latitude="lat")
    ds = ds.sel(**kwargs)
    return ds
