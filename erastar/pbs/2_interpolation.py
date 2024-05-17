import dask
import intake
import numpy as np
import pandas as pd
import xarray as xr
import dask_jobqueue
import dask.distributed

import os
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

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

def setup_cluster(
        jobs, 
        processes,
        journal,
        fraction=0.8,
        timeout=20, # en minute
        **kwargs,
    ):
    
    cluster = dask_jobqueue.PBSCluster(
        processes=processes,
        **kwargs
    )
    cluster.scale(jobs=jobs)
    client = dask.distributed.Client(cluster)

    if not processes:
        processes = cluster.worker_spec[0]["options"]["processes"]

    flag = True
    start = time.time()
    while flag:
        wk = client.scheduler_info()["workers"]
        journal.info(f"       ({int(time.time()- start)}s) - Number of workers up = {len(wk)}")
        time.sleep(5)
        if len(wk) >= processes * jobs * fraction:
            flag = False
            journal.info(f"     Cluster is up, proceeding with computations")
        now = time.time()
        if (now - start) / 60 > timeout:
            flag = False
            journal.info(f"     Timeout: cluster did not spin up, closing")
            cluster.close()
            client.close()
            cluster, client = None, None

    return cluster, client

def open_datacube(path, chunks, start, end):
    """Ouverture du datacube"""
    vars = ["es_u10s", "es_v10s", "e5_u10s", "e5_v10s", "count", "quality_flag"]
    cat = intake.open_catalog(path)
    datacubes = [cat[str(year)](chunks=chunks).to_dask().drop_vars(vars) for year in range(start, end+1)]
    datacube = xr.concat(datacubes, dim="time")

    datacube = (datacube.rename(
                {v: v.replace("tauu", "taue") for v in datacube if "tauu" in v}
            ).rename(
                {v: v.replace("tauv", "taun") for v in datacube if "tauv" in v}
            )
    )
    return datacube

def initialisation(ds, **kwargs):
    """Inialisation du dataset de matchup, afin de préparer l'étape de pré-traitement."""
    
    # Sélection des matchups avec une distance inférieur à 200km ? 
    ds = ds.where(ds_raw.alti___distance.compute()<2e5, drop=True)
    
    vars = [
        '__site_id','__site_name','alti___distance','alti___source','alti___source_center_index','alti___source_slices',
        'alti___time_difference','alti_cycle','alti_dac','alti_ggx_dac','alti_ggx_internal_tide','alti_ggx_mdt','alti_ggx_ocean_tide',
        'alti_ggx_sla_filtered','alti_ggx_sla_unfiltered','alti_ggx_sla_unfiltered_denoised','alti_ggx_sla_unfiltered_denoised_uncertainty',
        'alti_ggx_sla_unfiltered_imf1','alti_ggx_sla_unfiltered_noise','alti_internal_tide', 'alti_lwe','alti_mdt','alti_ocean_tide',
        'alti_sla_filtered','alti_sla_unfiltered','alti_sla_unfiltered_denoised','alti_sla_unfiltered_denoised_uncertainty','alti_sla_unfiltered_imf1',
        'alti_sla_unfiltered_noise','alti_tpa_correction','alti_track','box_latc','box_lonc','box_phi','box_theta_lat','box_theta_lon',
        'drifter_WMO','drifter_acc_x','drifter_acc_y','drifter_coriolis_x','drifter_coriolis_y','drifter_deploy_date','drifter_deploy_lat',
        'drifter_deploy_lon','drifter_drogue_lost_date','drifter_end_date','drifter_end_lat','drifter_end_lon','drifter_err_lat','drifter_err_lon',
        'drifter_err_ve','drifter_err_vn','drifter_expno','drifter_gap','drifter_lon360','drifter_rowsize','drifter_theta_lat','drifter_theta_lon',
        'drifter_typebuoy','drifter_typedeath','drifter_ve', 'drifter_vn','drifter_vx', 'drifter_vy','f','alti_time_mid', 
        'alti_x','alti_x_mid','alti_y','alti_y_mid']

    # Suppression des variables sans intérets
    ds = ds.drop_vars(vars)

    # Pour chaque observation, définition de l'extension temporelle
    ds['time_min'] = (ds.time + np.timedelta64(kwargs['dt'][0], 'h'))
    ds['time_max'] = (ds.time + np.timedelta64(kwargs['dt'][1]-1, 'h'))
    return ds

_lon_180_to_360 = lambda lon: lon % 360

def limite_lon(boxs, drifters):
    """ Fonction universel (ufunc) qui créé un mask pour les observations avec une longitude en dehors 
    de la plage [-178:178] """
    limite_lons = []
    for box, drifter in zip(boxs, drifters):
        limite_lon = ((drifter > 178).any() or (drifter < -178).any() or (box > 178).any() or (box < -178).any())
        if limite_lon: 
            limite_lons.append(True)
        else:
            limite_lons.append(False)
    return np.array(limite_lons)

def lon_180_to_360(values, limits):
    """ Fonction universelle (ufunc) qui pour chaque observation, convertis les valeurs de longitudes 
    de la plage [-180:180] à [0:360], si les valeurs de longitude sont en dehors de la plage [-178:178]. """
    new_values = []
    for value, limit in zip(values, limits):
        if limit:
            new_values.append(_lon_180_to_360(value))
        else:
            new_values.append(value)
    return np.array(new_values)

def lon_180_to_360_alti(values, limits, **kwargs):
    """ Fonction universelle (ufunc) qui pour chaque observation, convertis les valeurs de alti_lon 
    de la plage [-180:180] à [0:360], si les valeurs de longitude sont en dehors de la plage [-178:178]. """
    new_values = []
    for value, limit in zip(values, limits):
        if limit:
            new_values.append(_lon_180_to_360(value[kwargs['idx']]))
        else:
            new_values.append(value[kwargs['idx']])
    return np.array(new_values)

def mask_drifter_time(drifter_times, times, **kwargs):
    """ Fonction universelle (ufunc) qui pour chaque observation créé un mask pour les valeurs de drifter_time qui sont
    dans la plage +/- dt jour autour de la valeur de time. """
    values = []
    for drifter_time, time in zip(drifter_times, times):
        t = pd.to_datetime(time)
        start = np.datetime64((t + pd.Timedelta(hours=kwargs['dt'][0])))
        end = np.datetime64((t + pd.Timedelta(hours=kwargs['dt'][1]-1)))
        mask = (drifter_time >= start) & (drifter_time <= end)
        drifter_time[~mask] = np.datetime64('NaT')
        drifter_time.astype(np.int64)
        values.append(drifter_time)
    return np.array(values)

def set_time_array(times, **kwargs):
    """ Fonction universelle (ufunc) qui pour chaque observation, génère une liste de date qui sont dans 
    la plage +/- dt jour autour de la valeur de time. """
    values = []
    for time in zip(times):
        t = pd.to_datetime(time)
        t = [np.datetime64((t + pd.Timedelta(hours=_dt)).date[0]) for _dt in range(*kwargs['dt'])]
        values.append(t)
    return np.array(values).astype(np.int64)
    
def min_apply(boxs, drifters):
    """ Fonction universelle (ufunc) qui pour chaque observation recupère la valeur minimale de longitude/latitude entre box et drifter. """
    return np.array([min(box.min(), drifter.min()) for box, drifter in zip(boxs, drifters)])

def max_apply(boxs, drifters):
    """ Fonction universelle (ufunc) qui pour chaque observation recupère la valeur maximale de longitude/latitude entre box et drifter. """
    return np.array([max(box.max(), drifter.max()) for box, drifter in zip(boxs, drifters)])

def set_limite_lon(ds):
    # Créé un mask des observation avec une valeur de longitude (box_lon, drifter_lon) en dehors de la plage [-178:178]
    ds['limite_lon'] = xr.apply_ufunc(
        limite_lon, 
        ds["box_lon"], 
        ds["drifter_lon"],
        input_core_dims=[['box_x', 'box_y'], ['site_obs']],
        output_core_dims=[[]],
        output_dtypes=[bool],
        dask='parallelized'
    )
    return ds

def set_new_box_lon(ds):
    # Conversion des valeurs de box_lon de la plage [-180:180] à [0:360] pour les observations dont limite_lon = True
    ds['new_box_lon'] = xr.apply_ufunc(
        lon_180_to_360, 
        ds["box_lon"], 
        ds["limite_lon"],
        input_core_dims=[['box_y', 'box_x'], []],
        output_core_dims=[['box_y', 'box_x']],
        output_dtypes=[float],
        dask_gufunc_kwargs=dict(output_sizes={'box_y':80, 'box_x':120}),
        dask='parallelized'
    )
    return ds

def set_new_drifter_lon(ds):
    # Conversion des valeurs de drifter_lon de la plage [-180:180] à [0:360] pour les observations dont limite_lon = True
    ds['new_drifter_lon'] = xr.apply_ufunc(
        lon_180_to_360, 
        ds["drifter_lon"], 
        ds["limite_lon"],
        input_core_dims=[['site_obs'], []],
        output_core_dims=[['site_obs']],
        output_dtypes=[float],
        dask_gufunc_kwargs=dict(output_sizes={'site_obs': ds.sizes['site_obs']}),
        dask='parallelized'
    )
    return ds

def set_new_alti_lon(ds):
    # Conversion des valeurs de alti_lon de la plage [-180:180] à [0:360] pour les observations dont limite_lon = True
    ds['new_alti_lon'] = xr.apply_ufunc(
        lon_180_to_360_alti, 
        ds["alti_lon"], 
        ds["limite_lon"],
        input_core_dims=[['alti_time'], []],
        output_core_dims=[[]],
        output_dtypes=[float],
        dask='parallelized',
        kwargs={'idx': ds.sizes['alti_time']//2}
    )
    return ds

def set_new_drifter_time(ds, dt):
    # Sélection des valeurs de drifter_time qui sont dans la plage +/- dt jour autour de la valeur de time.
    ds['new_drifter_time'] = xr.apply_ufunc(
        mask_drifter_time, 
        ds["drifter_time"].chunk(dict(site_obs=-1)), 
        ds["time"],
        input_core_dims=[['site_obs'], []],
        output_core_dims=[['site_obs']],
        output_dtypes=[int],
        dask_gufunc_kwargs=dict(output_sizes={'site_obs': ds.sizes['site_obs']}),
        dask='parallelized',
        kwargs={'dt': dt} # +/- 2 jours
    ).astype('datetime64[ns]')
    return ds

def set_new_drifter_time_array(ds, dt):
    # Création d'une liste de date qui sont dans la plage +/- dt jour autour de la valeur de time.
    ds['_drifter_time_array'] = xr.apply_ufunc(
        set_time_array, 
        ds["time"],
        input_core_dims=[[]],
        output_core_dims=[['t']],
        output_dtypes=[int],
        dask_gufunc_kwargs=dict(output_sizes={'t': len(range(*dt))}),
        dask='parallelized',
        kwargs={'dt': dt} # +/- 2 jours
    ).astype('datetime64[ns]')
    return ds

def set_matchup_site(ds):
    # Récupération des valeurs de longitude, latitude et temps des bouées dérivantes
    ds_obs_m = ds.isel(site_obs=xr.DataArray(ds.__site_matchup_indice.values.astype(int), dims='obs'))
    ds['_drifter_lon'] = ds_obs_m['new_drifter_lon']
    ds['_drifter_lat'] = ds_obs_m['drifter_lat']
    ds['_drifter_time'] = ds_obs_m['drifter_time']
    return ds

def set_lon_min(ds):
    # Récupère la valeur de longitude minimale entre new_box_lon et new_drifter_lon
    ds['lon_min'] = xr.apply_ufunc(
        min_apply, 
        ds["new_box_lon"], 
        ds["new_drifter_lon"],
        input_core_dims=[['box_y', 'box_x'], ['site_obs']],
        output_core_dims=[[]],
        output_dtypes=[float],
        dask='parallelized',
    )
    return ds

def set_lon_max(ds):
    # Récupère la valeur de longitude maximale entre new_box_lon et new_drifter_lon
    ds['lon_max'] = xr.apply_ufunc(
        max_apply, 
        ds["new_box_lon"], 
        ds["new_drifter_lon"],
        input_core_dims=[['box_y', 'box_x'], ['site_obs']],
        output_core_dims=[[]],
        output_dtypes=[float],
        dask='parallelized',
    )
    return ds

def set_lat_min(ds):
    # Récupère la valeur de latitude minimale entre new_box_lat et new_drifter_lat
    ds['lat_min'] = xr.apply_ufunc(
        min_apply, 
        ds["box_lat"], 
        ds["drifter_lat"],
        input_core_dims=[['box_y', 'box_x'], ['site_obs']],
        output_core_dims=[[]],
        output_dtypes=[float],
        dask='parallelized',
    )
    return ds

def set_lat_max(ds):
    # Récupère la valeur de latitude maximale entre new_box_lat et new_drifter_lat
    ds['lat_max'] = xr.apply_ufunc(
        max_apply, 
        ds["box_lat"], 
        ds["drifter_lat"],
        input_core_dims=[['box_y', 'box_x'], ['site_obs']],
        output_core_dims=[[]],
        output_dtypes=[float],
        dask='parallelized',
    )
    return ds

def interp_traj(datacube, ds):
    # DRIFTER TRAJECTORY
    res_traj = (
        datacube
        .interp(
            time=ds.new_drifter_time,
            lon=ds['new_drifter_lon'],
            lat=ds['drifter_lat'],
            )
        .rename({v: "es_traj_" + v for v in datacube})
        .drop_vars(
            [
                "drifter_lon",
                "drifter_lat",
                "lon",
                "lat",
                "time",
            ]
        )
    )
    return res_traj

def interp_box(datacube, ds):
    # BOX
    res_box = (
        datacube
        .interp(
            time=ds._drifter_time,
            lon=ds.new_box_lon,
            lat=ds.box_lat,
        )
        .drop_vars(["lat", "lon", "time"])
        .rename({v: "es_box_" + v for v in datacube})
    )
    return res_box

def interp_drifter(datacube, ds):
    # DRIFTER MATCHUP POSITION
    res_drifter = (
        datacube
        .interp(
            time=ds['_drifter_time_array'],
            lon=ds['_drifter_lon'],
            lat=ds['_drifter_lat'],
        )
    )
    res_drifter['es_time_'] = ds['_drifter_time_array']
    res_drifter = (
        res_drifter.set_coords(['es_time_'])
        .drop_vars(["lat", "lon", "time"])
        .rename({v: "es_drifter_temp_" + v for v in datacube})
        .rename({'t': 'es_time'})
    )
    return res_drifter

def interp_alti(datacube, ds):
    # ALTI
    res_alti = (
        datacube
        .interp(
            time=ds['alti_time_'].isel(alti_time=ds.sizes['alti_time']//2),
            lon=ds['new_alti_lon'],
            lat=ds['alti_lat'].isel(alti_time=ds.sizes['alti_time']//2),
        )
    )
    res_alti = res_alti.drop_vars(list(res_alti.coords.keys())).rename(
        {v: "es_alti_matchup_" + v for v in datacube}
    )
    return res_alti

def merge_results(ds, res_traj, res_box, res_drifter, res_alti):
    res_drifter_matchup = (
        res_traj.isel(site_obs=xr.DataArray(ds.__site_matchup_indice.values.astype(int), dims='obs'))
        .drop_vars(list(res_traj.coords.keys()))
        .rename({v: v.replace("traj_", "drifter_matchup_") for v in res_traj})
    )

    res = (
        xr.merge([res_traj, res_drifter_matchup, res_box, res_drifter, res_alti])
            .reset_coords(["drifter_time", "drifter_x", "drifter_y", "es_time_"])
    )

    res = res.set_coords(
            ["obs", "es_time_"]
        )

    res = res.rename({v: "es_" + v.replace("es_", "") for v in res if "es_" in v})
    res = res.rename({v: "e5_" + v.replace("es_", "").replace("e5_", "") for v in res if "e5_" in v})
    return res

def set_attrs(res):
    
    # attrs
    res.e5_alti_matchup_taue.attrs = {
        "long_name": "era5 eastward wind stress interpolated on the altimeter's matchup",
        "units": "Pa",
    }
    res.e5_alti_matchup_taun.attrs = {
        "long_name": "era5 northward wind stress interpolated on the altimeter's matchup",
        "units": "Pa",
    }
    
    res.e5_drifter_matchup_taue.attrs = {
        "long_name": "era5 eastward wind stress interpolated on the drifter's matchup",
        "units": "Pa",
    }
    res.e5_drifter_matchup_taun.attrs = {
        "long_name": "era5 northward wind stress interpolated on the drifter's matchup",
        "units": "Pa",
    }
    
    res.e5_box_taue.attrs = {
        "long_name": "era5 eastward wind stress interpolated on the box at the matchup time or several times (depending on the dimension)",
        "units": "Pa",
    }
    res.e5_box_taun.attrs = {
        "long_name": "era5 northward wind stress interpolated on the box at the matchup time or several times (depending on the dimension)",
        "units": "Pa",
    }
    
    res.e5_drifter_temp_taue.attrs = {
        "long_name": "era5 eastward wind stress interpolated on the drifter's matchup position for several times (static but temporal variation)",
        "units": "Pa",
    }
    res.e5_drifter_temp_taun.attrs = {
        "long_name": "era5 northward wind stress interpolated on the drifter's matchup position for several times (static but temporal variation)",
        "units": "Pa",
    }
    
    res.e5_traj_taue.attrs = {
        "long_name": "era5 eastward wind stress interpolated on the drifter's trajectory",
        "units": "Pa",
    }
    res.e5_traj_taun.attrs = {
        "long_name": "era5 northward wind stress interpolated on the drifter's trajectory",
        "units": "Pa",
    }
    
    res.es_alti_matchup_taue.attrs = {
        "long_name": "erastar eastward wind stress interpolated on the altimeter's matchup",
        "units": "Pa",
    }
    res.es_alti_matchup_taun.attrs = {
        "long_name": "erastar northward wind stress interpolated on the altimeter's matchup",
        "units": "Pa",
    }
    
    res.es_drifter_matchup_taue.attrs = {
        "long_name": "erastar eastward wind stress interpolated on the drifter's matchup",
        "units": "Pa",
    }
    res.es_drifter_matchup_taun.attrs = {
        "long_name": "erastar northward wind stress interpolated on the drifter's matchup",
        "units": "Pa",
    }
    
    res.es_box_taue.attrs = {
        "long_name": "erastar eastward wind stress interpolated on the box at the matchup time",
        "units": "Pa",
    }
    res.es_box_taun.attrs = {
        "long_name": "erastar northward wind stress interpolated on the box at the matchup time",
        "units": "Pa",
    }
    
    res.es_drifter_temp_taue.attrs = {
        "long_name": "erastar eastward wind stress interpolated on the drifter's matchup position for several times (static but temporal variation)",
        "units": "Pa",
    }
    res.es_drifter_temp_taun.attrs = {
        "long_name": "erastar northward wind stress interpolated on the drifter's matchup position for several times (static but temporal variation)",
        "units": "Pa",
    }
    
    res.es_traj_taue.attrs = {
        "long_name": "erastar eastward wind stress interpolated on the drifter's trajectory",
        "units": "Pa",
    }
    res.es_traj_taun.attrs = {
        "long_name": "erastar northward wind stress interpolated on the drifter's trajectory",
        "units": "Pa",
    }
    
    return res

def interpolation(datacube, ds, is_360=False):
    """Interpolation avec le datacube"""
    _lon_180_to_360 = lambda lon: lon % 360
    
    if is_360 == True:
        datacube = datacube.assign_coords(lon=_lon_180_to_360(datacube.lon))
        datacube = datacube.sortby("lon")    
    
    res_traj = interp_traj(datacube, ds)
    res_box = interp_box(datacube, ds)
    res_drifter = interp_drifter(datacube, ds)
    res_alti = interp_alti(datacube, ds)
    res = merge_results(ds, res_traj, res_box, res_drifter, res_alti)
    res = set_attrs(res)
    return res

def preprocessing(ds, dt):
    """Pré-traitement du dataset des matchups"""
    # Preprocessing des données d'observations
    ds = set_limite_lon(ds)
    ds = set_new_box_lon(ds)
    ds = set_new_drifter_lon(ds)
    ds = set_new_alti_lon(ds)
    ds = set_new_drifter_time(ds, dt)
    ds = set_new_drifter_time_array(ds, dt)
    ds = set_matchup_site(ds)
    ds = set_lon_min(ds)
    ds = set_lon_max(ds)
    ds = set_lat_min(ds)
    ds = set_lat_max(ds)
    return ds

@dask.delayed
def processing(ds, datacube, dt):
    """Traitement entre les données du dataset des matchups et le datacube, 
    avec interpolation selon limite_lon = True or False et concaténation"""
    # Préparation des points d'observations (preprocessing)
    ds = preprocessing(ds, dt=dt)

    # Séparation des points avec limite_lon True or False (car traitement différents)
    ds_180 = ds.where(ds.limite_lon.compute() == False, drop=True)
    ds_360 = ds.where(ds.limite_lon.compute() == True, drop=True)

    # Interpolations et traitements
    res_180 = interpolation(datacube, ds_180)
    res_360 = interpolation(datacube, ds_360, is_360=True)

    # Concaténation des résultats pour limite_lon True or False
    res = xr.merge([res_180, res_360])

    return res

@dask.delayed
def concat(results, output, filename, fmt):
    """Concaténation des batches, et sauvegarde sur disque"""
    r = dask.compute(*results)
    res = xr.concat(r, "obs")
    
    if fmt == 'netcdf':
        res.to_netcdf(output / f'{filename}.nc')
    if fmt == 'zarr':
        res.to_zarr(output / f'{filename}.zarr')

def conf(ds, n_obs_per_batch, n_obs_per_file):
    """Visualisation des paramètres (nombre de fichiers de sorties, ..) en fonction de la configuration d'entrée"""
    n_batches = ds.sizes['obs']//n_obs_per_batch + (1 if ds.sizes['obs'] % n_obs_per_batch > 0 else 0) # Nombre de batch au total
    n_batch_per_file = n_obs_per_file // n_obs_per_batch
    n_netcdf = n_batches // n_batch_per_file + (1 if n_batches % n_batch_per_file > 0 else 0) # Nombre de fichier netCDF en sortie
    print(f"Nombre d'observation par batch: {n_obs_per_batch}")
    print(f"Nombre de batch: {n_batches}")
    print(f"Nombre d'observations par fichier NetCDF: {n_obs_per_file}")
    print(f"Nombre de batch par fichier netCDF: {n_batch_per_file}")
    print(f"Nombre de fichier NetCDF: {n_netcdf}")

def recipe(ds, datacube, dt, n_obs_per_file, n_obs_per_batch, n_obs_per_loop, idx_loop, output, filename, fmt):
    """Création des recettes en lazy_mode
    n_obs_per_batch : Nombre d'observations par batch
    n_obs_per_file : Nombre d'observations par fichier netCDF
    """

    batches = []
    for i in range(0, ds.sizes['obs'], n_obs_per_batch):
        # Sélection des observation dans la tranche
        ds_sub = ds.isel(obs=slice(i,i + n_obs_per_batch))
        time_min, time_max = ds_sub.time_min.values.min(), ds_sub.time_max.values.max()
        # Sélection de la portion du datacube concerné par la tranche
        datacube_sub = datacube.sel(time=slice(time_min, time_max))
        batches.append(processing(ds_sub, datacube_sub, dt))

    concats = []
    n_batch_per_file = n_obs_per_file // n_obs_per_batch
    for idx, i in enumerate(range(0, len(batches), n_batch_per_file)):
        r = batches[slice(i,i + n_batch_per_file)]

        n = n_obs_per_loop // n_obs_per_file + (1 if n_obs_per_loop % n_obs_per_file > 0 else 0)
        file = f'{filename}_{idx + idx_loop*n}'
        concats.append(concat(r, output, file, fmt))
    return concats

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='ERASTAR interpolation',
        description='Program to interpolate observations over ERASTAR datacube'
    )

    # PARAMETERS
    # parser.add_argument('--inputs', type=Path, default='./inputs.csv', help="path of input file with list of dataset observations paths")
    parser.add_argument('--dt', type=int, default=[-2, 3], nargs=2, help="days range")
    parser.add_argument('--n-obs-per-loop', type=int, default=150_000, help="Number of observations per loop")
    parser.add_argument('--n-obs-per-file', type=int, default=50_000, help="Number of observations per file")
    parser.add_argument('--n-obs-per-batch', type=int, default=1_000, help="Number of observations to be processed at the same time (Number of observations in a batch)")
    parser.add_argument('--format', type=str, default='netcdf', help="output file format (netcdf or zarr)")
    parser.add_argument('--inputs', type=Path, default='./inputs.csv', help="path of inputs file with list of matchup dataset paths")
    # DATACUBE
    parser.add_argument('--path-datacube', type=Path, required=True, help="path to the datacube reference.yaml file")
    # CLUSTER
    parser.add_argument('--queue', type=str, default='mpi', help='PBS queue list')
    parser.add_argument('--cores', type=int, default=28, help='list of the number of PBS cores per job')
    parser.add_argument('--jobs', type=int, default=2, help='list of the number of PBS jobs')
    parser.add_argument('--memory', type=str, default='115Gib', help='list of the number of PBS memory per job')
    parser.add_argument('--walltime', type=str, default='04:00:00', help='walltime for PBS jobs')
    parser.add_argument('--processes', type=int, help='list of the number of PBS processes per job')
    parser.add_argument('--interface', type=str, default='ib0', help='communication interface for PBS jobs')
    parser.add_argument('--log-directory', type=Path, required=True, help='path for cluster log directory')
    # OTHER
    parser.add_argument('--output-dir', type=Path, required=True, help='path for output files')
    args = parser.parse_args()

    # PATH
    output = create_dir(args.output_dir)

    # LOGGING
    journal = create_logger('journal', output / f'journal.log')         

    journal.info("DEBUT")
    start_time = time.time()

    # CLUSTER
    journal.info("  # CHARGEMENT DU CLUSTER")
    journal.info(f"     CLUSTER INFO: queue: {args.queue}, jobs: {args.jobs}, cores: {args.cores}, memory: {args.memory}, interface: {args.interface}, walltime: {args.walltime}")
    
    cluster = dask_jobqueue.PBSCluster
    
    kwargs = dict(
        queue=args.queue,
        cores=args.cores,
        memory=args.memory,
        walltime=args.walltime,    
        interface=args.interface,
        local_directory='/tmp',
        log_directory=args.log_directory, 
    )

    cluster, client = setup_cluster(
        args.jobs, 
        args.processes,
        journal,
        fraction=0.8,
        timeout=20, # en minute
        **kwargs,
    )

    port = client.scheduler_info()["services"]["dashboard"]
    ssh_command = f'ssh -N -L {port}:{os.environ["HOSTNAME"]}:8787 {os.environ["USER"]}@datarmor.ifremer.fr'

    journal.info(f"     {ssh_command}")
    journal.info(f"     open browser at address of the type: http://localhost:{port}") 

    # MATCHUPS
    df = pd.read_csv(args.inputs, sep=';')
    names = df[df['mask']]['name']

    for idx, name in enumerate(names):

        ## Ouverture des matchups
        journal.info(f"  # ({idx+1}/{len(names)}) {name}")
        journal.info(f"     1) Ouverture du fichier de matchups: {name}")

        zarr_dir = Path("/home/datawork-lops-osi/aponte/margot/historical_coloc")
        ds_raw = xr.open_zarr(zarr_dir / f'{name}.zarr')

        journal.info(f"        Nombre d'observations: {ds_raw.sizes['obs']}")

        ## Initialisation des matchups
        journal.info(f"     2) Initialisation des matchups") 
        ds = initialisation(ds_raw, dt=args.dt)

        ## Extension temporelle des matchups
        time_min, time_max = ds.time.values.min(), ds.time.values.max()
        dtime = np.timedelta64(max(abs(x) for x in args.dt), 'h')
        print(f"TIME_MIN: {time_min}, TIME_MAX: {time_max}")
        start = pd.to_datetime(time_min - dtime).year
        end = pd.to_datetime(time_max + dtime).year
        print(f"START: {start}, END: {end}")
        # Ouverture Datacube
        journal.info(f"     3) Ouverture du Datacube") 
    
        chunks = {"time": 1, "latitude": -1, "longitude": -1}
        datacube = open_datacube(args.path_datacube, chunks, start, end)
        journal.info(f"        Taille du datacube: {datacube.sizes}") 

        # Processing
        journal.info(f"     4) Processing par tranche de {args.n_obs_per_loop} observations") 
        for idx_loop, n_loop in enumerate(range(0, ds.sizes['obs'], args.n_obs_per_loop)):
            
            start_loop_time = time.time()

            # Sélection des observations dans la tranche
            ds_sub = ds.isel(obs=slice(n_loop,n_loop + args.n_obs_per_loop))

            # Création des recettes de la tranche
            journal.info(f"          {idx_loop+1}.1) Création des recettes de la tranche {n_loop}") 

            output_label = create_dir(output / name)
            filename = f"{name}_interp"
            concats = recipe(ds_sub, datacube, args.dt, args.n_obs_per_file, args.n_obs_per_batch, args.n_obs_per_loop, idx_loop, output_label, filename, args.format)

            journal.info(f"          {idx_loop+1}.2) Exécution des recettes de la tranche {n_loop}") 
            z = dask.compute(concats)
            end_loop_time = time.time()
            journal.info(f"             Temps d'éxecution: {np.round(end_loop_time-start_loop_time)}s") 

    end_time = time.time()
    journal.info(f"FIN - Temps d'éxecution: {np.round(end_time-start_time)}s")

