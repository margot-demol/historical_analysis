import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from dask_jobqueue import PBSCluster
from dask.distributed import Client
import intake

import utils
sys.path.insert(0, '../../..')
from histlib.observations import (
    generate_virtual_observations,
)

def sequential(datacube, ds, dt, dlon, dlat):
    """Select data cubes around observations in a larger datacube with sequential methode"""
    results = []
    for obs in ds.obs:
        
        start, end = obs.time + pd.Timedelta(days=-dt), obs.time + pd.Timedelta(days=dt)
        lon_min, lon_max = obs.lon - dlon, obs.lon + dlon
        lat_min, lat_max = obs.lat - dlat, obs.lat + dlat
        
        # Sélection d'un cube au sein du datacube
        results.append(
            datacube.sel(
                time=slice(start,end),
                longitude=slice(lon_min, lon_max),
                latitude=slice(lat_min, lat_max),
            ).compute()
        )

def get_index_array(ds, dt, dlon, dlat, dl):
    """Generate array of index for each observation"""
    # Génération des slices des cubes à partir des observations
    i_time, i_lon, i_lat = [], [] , []
    for obs in ds.obs:
        
        start, end = obs.time + pd.Timedelta(days=-dt), obs.time + pd.Timedelta(days=dt+1)
        lon_min, lon_max = obs.lon - dlon, obs.lon + dlon
        lat_min, lat_max = obs.lat - dlat, obs.lat + dlat
        
        s_time = slice(start.values, end.values, timedelta(days=1))
        s_lon = slice(lon_min, lon_max, dl)
        s_lat = slice(lat_min, lat_max, dl)
        
        i_time.append(np.r_[s_time])
        i_lon.append(np.r_[s_lon])
        i_lat.append(np.r_[s_lat])
    return i_time, i_lon, i_lat

def get_flatten_array(i_time, i_lon, i_lat):
    """Generate flatten array of index of all observation"""
    # Génération des listes d'indices à plat à partir des slices
    f_time, f_lat, f_lon = [], [], []
    pos, shapes = [], []
    n=0
    for it, ila, ilo in zip(i_time, i_lat, i_lon):
        
        # Créer des grilles multidimensionnelles
        gt, gla, glo = np.meshgrid(it, ila, ilo, indexing='ij')

        # Mise à plat des indices
        ft = gt.flatten()
        fla = gla.flatten()
        flo = glo.flatten()

        # Ajouter les indices à plat du cube, dans la liste des indices à plat
        f_time.extend(ft)
        f_lat.extend(fla)
        f_lon.extend(flo)

        # Enregistre la dimension du cube
        shapes.append((len(it), len(ila), len(ilo)))
        
        # Enregistre les indices correspondant au cube
        n1 = n + len(ft)
        pos.append(slice(n, n1))       
        n = n1
    return f_time, f_lon, f_lat 

def flatten(datacube, f_time, f_lon, f_lat):
    """Select data cubes around observations in a larger datacube with flatten methode"""

    result = datacube.sel(
        time=xr.DataArray(f_time,  dims="points"),
        longitude=xr.DataArray(f_lon,  dims="points"),
        latitude=xr.DataArray(f_lat,  dims="points"),
        method="nearest",
    ).compute()

def benchmark(method, dt, dlon, dlat, dl,
              number_obs, extent, period, 
              time_chunk, geo_chunk, datacube_size, 
              queue, jobs, cores, memory, processes, interface, walltime,
              output_dir, str_uuid):
    
    extent = tuple(extent)
    period = tuple(period)

    # PATH
    logs_dir = utils.create_dir(output_dir / 'logs')
    jrnl_dir = utils.create_dir(output_dir / 'journal')
    
    # LOGGING
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    journal = logging.getLogger('journal')
    journal.setLevel(logging.INFO)
    jrnl_handler = logging.FileHandler(jrnl_dir / f"{str_uuid}.log", mode='w')
    jrnl_handler.setFormatter(formatter)
    journal.addHandler(jrnl_handler)

    files = logging.getLogger('fsspec.local')
    files.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(logs_dir / f"{str_uuid}.log", mode='w')
    file_handler.setFormatter(formatter)
    files.addHandler(file_handler)

    try:
        # CLUSTER
        journal.info(f"CLUSTER INFO: queue: {queue}, jobs: {jobs}, cores: {cores}, memory: {memory}, processes: {processes}, interface: {interface}, walltime: {walltime}")
        cluster = PBSCluster(
            queue=queue, # Nécéssaire sinon erreur : qsub: Job rejected by all possible destinations
            cores=cores,
            processes=processes,       
            memory=memory,    
            interface=interface,
            walltime=walltime,
        )
        cluster.scale(jobs=jobs)
        client = Client(cluster)

        env = os.environ
        port = client.scheduler_info()["services"]["dashboard"]
        ssh_command = f'ssh -N -L {port}:{env["HOSTNAME"]}:8787 {env["USER"]}@datarmor.ifremer.fr'

        journal.info(f"    dashboard via ssh: {ssh_command}")
        journal.info(f"    open browser at address of the type: http://localhost:{port}")

        # DATACUBE
        journal.info(f"DATACUBE INFO: time-chunk: {time_chunk}, geo-chunk: {geo_chunk}, datacube-size: {datacube_size}")
        WRK = Path('/home1/datawork/gcaer/data/aviso')
        datacube = WRK / f"datacube-{datacube_size}/references/aviso/reference.yaml"
        cat = intake.open_catalog(datacube)
        # Liste des variables à supprimer
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
        datacube = cat["data"](chunks={"time": time_chunk, "latitude": geo_chunk, "longitude": geo_chunk}).to_dask().drop_vars(_drop)

        # OBSERVATIONS
        journal.info(f"OBSERVATION INFO: number-obs: {number_obs}, extent: {extent}, period: {period}")
        ds = generate_virtual_observations(number_obs, extent, period)

        # COLOCALISATIONS
        journal.info(f"COLOCALISATION INFO: method: {method}")
        start_all = time.time()
        if method == 'sequential':

            sequential(datacube, ds, dt, dlon, dlat)

        elif method == 'flatten':

            journal.info(f"    get_index_array")
            i_time, i_lon, i_lat = get_index_array(ds, dt, dlon, dlat, dl)

            journal.info(f"    get_flatten_array")
            f_time, f_lon, f_lat = get_flatten_array(i_time, i_lon, i_lat)
            journal.info(f"         flatten_array_length: {len(f_time)}")
            
            journal.info(f"    flatten")
            flatten(datacube, f_time, f_lon, f_lat)

        else:
            raise ValueError("La méthode {method} n'est pas une méthode valide. Choisissez parmi les méthodes suivantes: [sequentiel, flatten]")

        end_all = time.time()

        run_time_total = np.round(end_all - start_all,3) 

        journal.info(f"RESULT: run-time: {run_time_total}")

        client.close()
        cluster.close()

        journal.info(f"SUCCESS")
    except:
        journal.info(f"FAILED")
        run_time_total = None

    journal.info(f"FIN")

    file_handler.close()
    files.handlers.pop()

    jrnl_handler.close()
    journal.handlers.pop()

    return run_time_total




