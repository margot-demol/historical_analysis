import os
import yaml
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
path_format = (
    "/home/ref-cmems-public/tac/sea-level/SEALEVEL_GLO_PHY_L4_MY_008_047/"
    "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D/"
    "{time.year:04d}/{time.dayofyear:03d}/dt_global_allsat_phy_l4_{time:%Y%m%d}_*.nc"
)

def create_dir(path):
    """Créer le chemin de dossier 'path' s'il n'existe pas"""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path

def create_logger(name, path, format="%(asctime)s - %(message)s", level='INFO', mode='w'):
    """Créer un logger"""
    logger = logging.getLogger(name)
    handle = logging.FileHandler(path, mode=mode)
    formatter = logging.Formatter(format)
    handle.setFormatter(formatter)
    logger.addHandler(handle)
    logger.setLevel(level)
    return logger

def create_recipe(name, dates, path_format, output):
    
    def make_path(time):
        return glob(path_format.format(time=time))[0]

    time_concat_dim = ConcatDim("time", dates)
    pattern = FilePattern(make_path, time_concat_dim)
    recipe = HDFReferenceRecipe(pattern)
    
    # Création des dossiers
    target_path = output / 'references' / name 
    metadata_path = output / 'metadata' / name
    cache_path = output / 'cache' / name
    
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
    
    return recipe

def execute_recipe(recipe):
    task = recipe.to_dask()
    task.compute()

def create_recipes(groups, output):
    recipes = dict()
    for k, v in groups.items():
        output_year = output / str(k)
        recipes[k] = create_recipe(name, v, path_format, output_year)
    return recipes

def execute_recipes(recipes, journal):
    for k, v  in recipes.items():
        journal.info(f"    {k}")
        execute_recipe(v)

def create_catalog(groups, output):
    # Création du catalogue intake
    sources = dict()
    for year in groups.keys():
        sources[str(year)] = {
            'args': {
                'chunks': {},
                'consolidated': False,
                'storage_options': {
                    'fo': f'{output.as_posix()}/{year}/references/{name}/reference.json',
                    'remote_options': {},
                    'remote_protocol': 'file',
                    'skip_instance_cache': True,
                    'target_options': {},
                    'target_protocol': 'file'
                },
                'urlpath': 'reference://'
            },
            'description': '',
            'driver': 'intake_xarray.xzarr.ZarrSource'
        }

    config = {'sources':sources}
    # Convertir les données en format YAML
    yaml_data = yaml.dump(config)

    # Écrire les données YAML dans un fichier
    with open(output / 'reference.yaml', 'w') as fichier_yaml:
        fichier_yaml.write(yaml_data)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='AVISO generate datacube',
        description='Program to generate AVISO datacube'
    )

    # PARAMETERS
    parser.add_argument('--start', type=str, required=True, help="start date of datacube (format: '2000-01-01')")
    parser.add_argument('--end', type=str, required=True, help="end date of datacube (format: '2020-01-01')")
    parser.add_argument('--version', type=str, default='test', help="name of the version of the datacube (ex: datacube-2019)")
    # OTHER
    parser.add_argument('--output-dir', type=Path, required=True, help='path for output files')
    args = parser.parse_args()

    # PATH
    output = create_dir(args.output_dir / name / args.version)

    # # LOGGING
    journal = create_logger('journal', output / f'journal.log')         

    start = time.time()

    journal.info("DEBUT")
    journal.info("  1) Création de la recipe")

    dates = pd.date_range(args.start, args.end, freq="d")
    groups = dates.groupby(dates.year)

    recipes = create_recipes(groups, output)

    journal.info("  2) Chargement du cluster")

    cluster = LocalCluster()
    client = Client(cluster)

    port = client.scheduler_info()["services"]["dashboard"]
    ssh_command = f'ssh -N -L {port}:{os.environ["HOSTNAME"]}:8787 {os.environ["USER"]}@datarmor.ifremer.fr'

    journal.info(f"     {ssh_command}")
    journal.info(f"     open browser at address of the type: http://localhost:{port}") 

    journal.info("  3) Exécution de la recipe")

    execute_recipes(recipes, journal)

    create_catalog(groups, output)

    end = time.time()
    journal.info(f"FIN - Temps d'éxecution: {np.round(end-start)}s")