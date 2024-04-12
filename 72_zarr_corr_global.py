# --------------------- application related librairies and parameters ------------------

import os, shutil
import logging
from time import sleep, time

import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta, datetime

from dask import compute

# from dask.delayed import delayed
from dask.distributed import performance_report, wait
from distributed.diagnostics import MemorySampler

# to force flushing of memory
import gc, ctypes

import os
from glob import glob

import histlib.matchup as match
import histlib.diagnosis as diag
from histlib.cstes import labels, zarr_dir, matchup_dir, cutoff_str
from histlib.matchup import _data_var, _stress_var, _aviso_var

# ---- Run parameters

#root_dir = "/home/datawork-lops-osi/equinox/mit4320/parcels/"
run_name = "ms_global"


# dask parameters

dask_jobs = 1  # number of dask pbd jobs
jobqueuekw = dict(processes=20, cores=20)  # uplet debug

# ---------------------------- dask utils - do not touch -------------------------------


def spin_up_cluster(
    ctype,
    jobs=None,
    processes=None,
    fraction=0.8,
    timeout=20,
    **kwargs,
):
    """Spin up a dask cluster ... or not
    Waits for workers to be up for distributed ones

    Paramaters
    ----------
    ctype: None, str
        Type of cluster: None=no cluster, "local", "distributed"
    jobs: int, optional
        Number of PBS jobs
    processes: int, optional
        Number of processes per job (overrides default in .config/dask/jobqueue.yml)
    timeout: int
        Timeout in minutes is cluster does not spins up.
        Default is 20 minutes
    fraction: float, optional
        Waits for fraction of workers to be up

    """

    if ctype is None:
        return
    elif ctype == "local":
        from dask.distributed import Client, LocalCluster

        dkwargs = dict(n_workers=14, threads_per_worker=1)
        dkwargs.update(**kwargs)
        cluster = LocalCluster(**dkwargs)  # these may not be hardcoded
        client = Client(cluster)
    elif ctype == "distributed":
        from dask_jobqueue import PBSCluster
        from dask.distributed import Client

        assert jobs, "you need to specify a number of dask-queue jobs"
        cluster = PBSCluster(processes=processes, **kwargs)
        cluster.scale(jobs=jobs)
        client = Client(cluster)

        if not processes:
            processes = cluster.worker_spec[0]["options"]["processes"]

        flag = True
        start = time()
        while flag:
            wk = client.scheduler_info()["workers"]
            print("Number of workers up = {}".format(len(wk)))
            sleep(5)
            if len(wk) >= processes * jobs * fraction:
                flag = False
                print("Cluster is up, proceeding with computations")
            now = time()
            if (now - start) / 60 > timeout:
                flag = False
                print("Timeout: cluster did not spin up, closing")
                cluster.close()
                client.close()
                cluster, client = None, None

    return cluster, client


def dashboard_ssh_forward(client):
    """returns the command to execute on a local computer in order to
    have access to dashboard at the following address in a browser:
    http://localhost:8787/status
    """
    env = os.environ
    port = client.scheduler_info()["services"]["dashboard"]
    return f'ssh -N -L {port}:{env["HOSTNAME"]}:8787 {env["USER"]}@datarmor1-10g', port


def close_dask(cluster, client):
    logging.info("starts closing dask cluster ...")
    try:

        client.close()
        logging.info("client closed ...")
        # manually kill pbs jobs
        manual_kill_jobs()
        logging.info("manually killed jobs ...")
        # cluster.close()
        # logging.info("cluster closed ...")
    except:
        logging.exception("cluster.close failed ...")
        # manually kill pbs jobs
        manual_kill_jobs()

    logging.info("... done")


def manual_kill_jobs():
    """manually kill dask pbs jobs"""

    import subprocess, getpass

    #
    username = getpass.getuser()
    #
    bashCommand = "qstat"
    try:
        output = subprocess.check_output(
            bashCommand, shell=True, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "command '{}' return with error (code {}): {}".format(
                e.cmd, e.returncode, e.output
            )
        )
    #
    for line in output.splitlines():
        lined = line.decode("UTF-8")
        if username in lined and "dask" in lined:
            pid = lined.split(".")[0]
            bashCommand = "qdel " + str(pid)
            logging.info(" " + bashCommand)
            try:
                boutput = subprocess.check_output(bashCommand, shell=True)
            except subprocess.CalledProcessError as e:
                # print(e.output.decode())
                pass


def dask_compute_batch(computations, client, batch_size=None):
    """breaks down a list of computations into batches"""
    # compute batch size according to number of workers
    if batch_size is None:
        # batch_size = len(client.scheduler_info()["workers"])
        batch_size = sum(list(client.nthreads().values()))
    # find batch indices
    total_range = range(len(computations))
    splits = max(1, np.ceil(len(total_range) / batch_size))
    batches = np.array_split(total_range, splits)
    # launch computations
    outputs = []
    for b in batches:
        logging.info("batches: " + str(b) + " / " + str(total_range))
        out = compute(*computations[slice(b[0], b[-1] + 1)])
        outputs.append(out)

        # try to manually clean up memory
        # https://coiled.io/blog/tackling-unmanaged-memory-with-dask/
        client.run(gc.collect)
        client.run(trim_memory)  # should not be done systematically

    return sum(outputs, ())


def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


# ---------------------------------- core of the job to be done ----------------------------------
DL =25e3 #meters
DT = 0.5*3600 #seconds

#IF restricted computation
wd_x= ['es_cstrio_z15_alti_wd_x', 'es_cstrio_z15_drifter_wd_x']
wd_y = []
grad_x = ['alti_ggx_adt_filtered',
          'alti_ggx_adt_filtered_ocean_tide',
          'alti_ggx_adt_filtered_ocean_tide_internal_tide',
          'alti_ggx_adt_filtered_ocean_tide_internal_tide_dac',
         'aviso_alti_ggx_adt',
          'aviso_drifter_ggx_adt'
         ]
grad_y =[]
cutoff=[0, 0.1, 0.2, 0.5, 1, 1.5, 2, 2.5]

var = wd_x+grad_x+grad_y+['drifter_acc_x_0','drifter_coriolis_x_0']


def corr_dataset(dsm, l, wd_x=wd_x, wd_y=wd_y, grad_x=grad_x, grad_y=grad_y, cutoff=cutoff) :
    """ Compute the mean of product \alpha_i\alpha_j, giving correlation"""
    ds_ = match.product_dataset(dsm, wd_x=wd_x, wd_y=wd_y, grad_x=grad_x, grad_y=grad_y, cutoff=cutoff)   
    nb = ds_.dims["obs"]
    ds = ds_.mean('obs')
    ds["nb_coloc"] = nb
    ds['drifter_sat_year']=l
    ds = ds.expand_dims('drifter_sat_year')
    ds = ds.set_coords('drifter_sat_year')

    ## ATTRS
    for v in ds_.keys() :
        ds[v].attrs =ds_[v].attrs
        ds[v].attrs['units']=r'$m^2.s^{-4}$'
        
    return ds

def corr_var_dataset(dsm, l, wd_x=wd_x, wd_y=wd_y, grad_x=grad_x, grad_y=grad_y, cutoff=cutoff) :
    """ Compute the variance of product \alpha_i\alpha_j, then will allow to compute errors on correlation estimation"""
    ds_ = match.product_dataset(dsm, wd_x=wd_x, wd_y=wd_y, grad_x=grad_x, grad_y=grad_y, cutoff=cutoff)   
    nb = ds_.dims["obs"]
    ds = ds_.var('obs')
    ds["nb_coloc"] = nb
    ds['drifter_sat_year']=l
    ds = ds.expand_dims('drifter_sat_year')
    ds = ds.set_coords('drifter_sat_year')

    ## ATTRS
    for v in ds_.keys() :
        ds[v].attrs =ds_[v].attrs
        ds[v].attrs['units']=r'$m^2.s^{-4}$'
    return ds

def run_corr_global(l):

    dsm = xr.open_dataset(os.path.join(matchup_dir, f'matchup_{l}.zarr'))[var+['drogue_status', 'alti___distance', 'alti___time_difference']].dropna('obs').chunk({'obs':500})
    # add filtered acc and coriolis
    zarr_cutoff = os.path.join(zarr_dir+'_ok','cutoff_matchup',"cutoff_matchup_"+l+".zarr")
    zarr_cutoff1 = os.path.join(zarr_dir+'_ok','cutoff_matchup',"cutoff_matchup_"+l+"_2.zarr")
    if os.path.isdir(zarr_cutoff):
        dsm = xr.merge([dsm, xr.open_dataset(zarr_cutoff),xr.open_dataset(zarr_cutoff1)], join='inner').chunk({'obs':1000})
        
    dsm = dsm.where(dsm.alti___distance<=DL, drop=True)
    dsm = dsm.where(dsm.alti___time_difference<=DT, drop=True).drop(['alti___distance', 'alti___time_difference']).dropna('obs')
    dsmd = dsm.where(dsm.drogue_status, drop=True).drop('drogue_status')
    dsmnd = dsm.where(np.logical_not(dsm.drogue_status), drop=True).drop('drogue_status')
    dsm = dsm.drop('drogue_status')
    
    if l==labels[0] : 
        corr_dataset(dsm, l).to_zarr(os.path.join(zarr_dir+'_ok','global',f'corr_{int(DL//1000)}_{DT}.zarr'),encoding={'drifter_sat_year':{'dtype':'U32'}}, mode='w')
        if dsmd.dims['obs']!=0 : 
            corr_dataset(dsmd,l).to_zarr(os.path.join(zarr_dir+'_ok','global',f'corr_{int(DL//1000)}_{DT}_drogued.zarr'), encoding={'drifter_sat_year':{'dtype':'U32'}},mode='w')
        if dsmnd.dims['obs']!=0 : 
            corr_dataset(dsmnd,l).to_zarr(os.path.join(zarr_dir+'_ok','global',f'corr_{int(DL//1000)}_{DT}_undrogued.zarr'),encoding={'drifter_sat_year':{'dtype':'U32'}}, mode='w')
    else : 
        corr_dataset(dsm,l).to_zarr(os.path.join(zarr_dir+'_ok','global',f'corr_{int(DL//1000)}_{DT}.zarr'), append_dim='drifter_sat_year')
        if dsmd.dims['obs']!=0 : 
            corr_dataset(dsmd,l).to_zarr(os.path.join(zarr_dir+'_ok','global',f'corr_{int(DL//1000)}_{DT}_drogued.zarr'), append_dim='drifter_sat_year')
        if dsmnd.dims['obs']!=0 : 
            corr_dataset(dsmnd,l).to_zarr(os.path.join(zarr_dir+'_ok','global',f'corr_{int(DL//1000)}_{DT}_undrogued.zarr'),append_dim='drifter_sat_year')
    
if __name__ == "__main__":

    ## step0: setup logging

    # to std output
    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    # to file
    logging.basicConfig(
        filename="distributed.log",
        level=logging.INFO,
        # level=logging.DEBUG,
    )
    # level order is: DEBUG, INFO, WARNING, ERROR
    # encoding='utf-8', # available only in latests python versions

    # dask memory logging:
    # https://blog.dask.org/2021/03/11/dask_memory_usage
    # log dask summary?

    # spin up cluster
    logging.info("start spinning up dask cluster, jobs={}".format(dask_jobs))
    cluster, client = spin_up_cluster(
        "distributed",
        jobs=dask_jobs,
        fraction=0.9,
        walltime="08:00:00",
        **jobqueuekw,
    )
    ssh_command, dashboard_port = dashboard_ssh_forward(client)
    logging.info("dashboard via ssh: " + ssh_command)
    logging.info(f"open browser at address of the type: http://localhost:{dashboard_port}")


    ## boucle for on labels
    for l in labels: 
        logging.info(f"start processing {l}")
        run_corr_global(l)
        logging.info(f"end processing {l}")
    
    # close dask
    close_dask(cluster, client)

    logging.info("- all done")