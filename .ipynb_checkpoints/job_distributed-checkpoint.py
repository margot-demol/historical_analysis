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

import histlib.box as box
import histlib.aviso as aviso
import histlib.erastar as eras
from histlib.cstes import labels, zarr_dir, nc_files

dataset = 'eras'
# ---- Run parameters

#root_dir = "/home/datawork-lops-osi/equinox/mit4320/parcels/"
run_name = "box_build_colocalisations"

# will overwrite existing results
# overwrite = True
overwrite = False

# dask parameters

if dataset == 'coloc' :
    dask_jobs = 8  # number of dask pbd jobs
    jobqueuekw = dict(processes=3, cores=3)  # uplet debug

if dataset == 'aviso' :
    dask_jobs = 20  # number of dask pbd jobs for gps_sentinel_3A_2019
    jobqueuekw = dict(processes=15, cores=15)  # uplet debug
    
if dataset == 'eras' :
    dask_jobs = 10  # number of dask pbd jobs
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


def run_coloc(l, cluster, client):
    """main execution code"""

    # load or create dataset and do work
    # ds = ...
    # synthetic example here to compute pi:
    import dask.array as da

    L = [box.build_dataset(nc_files[l][i], persist=True) for i in range(len(nc_files[l]))]
    ds = (xr.concat(L, "obs")
      .assign_attrs(__time_coverage_end=xr.open_dataset(nc_files[l][-1]).attrs['__time_coverage_end'])
      .drop_vars('site_obs')
      .chunk(dict(obs=500))
     )
    #store
    zarr = os.path.join(zarr_dir, l+".zarr")
    ds.to_zarr(zarr, mode="w")
    logging.info(f"{l} storred in {zarr}")

def run_aviso(l, cluster, client):
    """main execution code"""

    # load or create dataset and do work
    # ds = ...
    # synthetic example here to compute pi:
    import dask.array as da
    ds_data = xr.open_zarr(zarr_dir+'/'+l+'.zarr')
    ds_data = ds_data.where(ds_data.alti___distance<2e5, drop=True).chunk({'obs':250, 'alti_time':-1, 'site_obs':-1})
    ds_aviso = aviso.compute_aviso_sla(ds_data).persist()
    #store
    zarr = os.path.join(zarr_dir, "aviso_"+l+".zarr")
    ds_aviso.to_zarr(zarr, mode="w")
    logging.info(f"aviso {l} storred in {zarr}")

def run_eras(l, cluster, client):
    """main execution code"""

    # load or create dataset and do work
    # ds = ...
    # synthetic example here to compute pi:
    import dask.array as da
    ds_data = xr.open_zarr(zarr_dir+'/'+l+'.zarr')
    ds_data = ds_data.where(ds_data.alti___distance<2e5, drop=True).chunk({'obs':200, 'alti_time':-1, 'site_obs':-1})
    ds_es = eras.compute_eras(ds_data, dt=(-12,13), only_matchup_time = True).chunk({'obs':500, 'site_obs':-1}).persist()
    #store
    zarr = os.path.join(zarr_dir, "erastar_"+l+".zarr")
    ds_es.to_zarr(zarr, mode="w")  
    logging.info(f"erastar {l} storred in {zarr}")
    
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
        walltime="4:00:00",
        **jobqueuekw,
    )
    ssh_command, dashboard_port = dashboard_ssh_forward(client)
    logging.info("dashboard via ssh: " + ssh_command)
    logging.info(f"open browser at address of the type: http://localhost:{dashboard_port}")

    #overwrite
    if not overwrite :
        if dataset == 'coloc' : labels = [l for l in labels if not os.path.isdir(os.path.join(zarr_dir, l+".zarr"))]
        if dataset == 'aviso' : labels = [l for l in labels if not os.path.isdir(os.path.join(zarr_dir, "aviso_"+l+".zarr"))]
        if dataset == 'eras' : labels = [l for l in labels if not os.path.isdir(os.path.join(zarr_dir, "erastar_"+l+".zarr"))]
    ## boucle for on labels
    for l in labels : 
        logging.info(f"start processing {l}")
        if dataset == 'coloc' : run_coloc(l, cluster, client)
        if dataset == 'aviso' : run_aviso(l, cluster, client)
        if dataset == 'eras' : run_eras(l, cluster, client)
        logging.info(f"end processing {l}")
    
    # close dask
    close_dask(cluster, client)

    logging.info("- all done")