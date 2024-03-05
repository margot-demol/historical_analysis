# --start 1993-01-01
# --end 2023-06-07
# --output-dir /home1/datawork/gcaer/data/aviso/datacube
# les données vont du 01/01/1993 au 07/06/2023

import os
import fsspec
import argparse
import time
import logging
import xarray as xr
import pandas as pd
import numpy as np
from glob import glob

from pangeo_forge_recipes.patterns import FilePattern, ConcatDim
from pangeo_forge_recipes.recipes import HDFReferenceRecipe
from pangeo_forge_recipes.storage import (
    CacheFSSpecTarget,
    FSSpecTarget,
    MetadataTarget,
    StorageConfig,
)
from dask.distributed import Client, LocalCluster
from pathlib import Path

name = "aviso"
data_root = "/home/ref-cmems-public/tac/sea-level/SEALEVEL_GLO_PHY_L4_MY_008_047/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D"

PATH_FORMAT = (
    "/home/ref-cmems-public/tac/sea-level/SEALEVEL_GLO_PHY_L4_MY_008_047/"
    "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D/"
    "{time.year:04d}/{time.dayofyear:03d}/dt_global_allsat_phy_l4_{time:%Y%m%d}_*.nc"
)

def make_path(time):
    return glob(PATH_FORMAT.format(time=time))[0]

# Logging
def create_logger(name, path, format="%(asctime)s - %(message)s", level='INFO'):
    logger = logging.getLogger(name)
    handle = logging.FileHandler(path, mode='w')
    formatter = logging.Formatter(format)
    handle.setFormatter(formatter)
    logger.addHandler(handle)
    logger.setLevel(level)
    return logger

if __name__ == "__main__":

    # Arguments
    def list_of_ints(arg):
        return list(map(int, arg.split(',')))

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str)
    parser.add_argument('--end', type=str)
    parser.add_argument('--output-dir', type=Path)

    args = parser.parse_args()

    # Path
    def create_dir(path):
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    output_dir = create_dir(args.output_dir)

    # Logging 
    journal = create_logger('journal', args.output_dir / 'journal.log')
    journal.info("DEBUT")
    journal.info("Création du FilePattern")

    start = time.time()
    
    dates = pd.date_range(args.start, args.end, freq="d")

    time_concat_dim = ConcatDim("time", dates)
    pattern = FilePattern(make_path, time_concat_dim)

    journal.info("Création de la Recipe")

    recipe = HDFReferenceRecipe(pattern)

    journal.info("Configuration de la Recipe")

    target_path = args.output_dir / f"references/{name}"
    metadata_path = args.output_dir / f"metadata/{name}"
    cache_path = args.output_dir / f"cache/{name}"

    fs, _, _ = fsspec.get_fs_token_paths(target_path)
    fs.mkdirs(target_path, exist_ok=True)
    target = FSSpecTarget(fs=fs, root_path=target_path)


    fs, _, _ = fsspec.get_fs_token_paths(metadata_path)
    if fs.exists(metadata_path):
        fs.rm(metadata_path, recursive=True)
    fs.mkdirs(metadata_path, exist_ok=True)
    metadata = MetadataTarget(fs=fs, root_path=metadata_path)


    fs, _, _ = fsspec.get_fs_token_paths(cache_path)
    if fs.exists(cache_path):
        fs.rm(cache_path, recursive=True)
    fs.mkdirs(cache_path, exist_ok=True)
    cache = CacheFSSpecTarget(fs=fs, root_path=cache_path)

    recipe.storage_config = StorageConfig(target=target, cache=cache, metadata=metadata)

    journal.info("Chargement du cluster")

    cluster = LocalCluster()
    client = Client(cluster)

    env = os.environ
    port = client.scheduler_info()["services"]["dashboard"]

    ssh_command = f'ssh -N -L {port}:{env["HOSTNAME"]}:8787 {env["USER"]}@datarmor.ifremer.fr'

    journal.info("dashboard via ssh: " + ssh_command)
    journal.info(f"open browser at address of the type: http://localhost:{port}") 

    task = recipe.to_dask()

    journal.info("Execution de la recipe")
    
    task.compute()

    end = time.time()
    
    journal.info(f"Temps d'éxecution: {np.round(end-start)}s")
    journal.info("FIN")