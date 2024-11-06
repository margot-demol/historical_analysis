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
from histlib.cstes import labels, zarr_dir

# ---- Run parameters

#root_dir = "/home/datawork-lops-osi/equinox/mit4320/parcels/"
run_name = "box_build_colocalisations"

# will overwrite existing results
#overwrite = True
overwrite = False

# dask parameters

dask_jobs = 2  # number of dask pbd jobs
jobqueuekw = dict(processes=10, cores=10)  # uplet debug

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
from histlib.matchup import add_adt_to_ds_data, add_low_pass_filter_to_data
cutoff = [3, 3.5]
chunk = 2000

def cutoff_matchup(l,T, cutoff):
    var = ['__site_matchup_indice', 'obs', 'box_x', 'box_y', 'alti_time_mid','f', "drifter_vx", "drifter_vy", "drifter_acc_x_0", "drifter_acc_y_0", "drifter_coriolis_x_0", "drifter_coriolis_y_0",]
    ds_data = xr.open_zarr(os.path.join(zarr_dir, f'{l}.zarr')).chunk({'obs':chunk, 'site_obs':-1})
    ds_data = ds_data.where(ds_data.alti___distance<2e5, drop=True).chunk({'obs':chunk, 'site_obs':-1})
    ds_data = add_adt_to_ds_data(ds_data)
    drogue_status = ds_data.time<ds_data.drifter_drogue_lost_date.mean('site_obs')
    ds_data = ds_data[var]

    add_low_pass_filter_to_data(ds_data, T=T, cutoff=cutoff)

    # SELECT MATCHUP
    cc = [l for l in list(ds_data.coords)if l not in ['obs', 'box_x', 'box_y', 'alti_time_mid']]
    ds_data = ds_data.reset_coords(cc)
    idx = ds_data.__site_matchup_indice.astype(int).compute()
    dsmf = ds_data.sel(site_obs=idx).isel( alti_time_mid=ds_data.dims['alti_time_mid']//2).drop(["alti_time_mid", "alti_x_mid", "alti_y_mid"])[[v for v in ds_data if 'drifter_acc_x_' in v or 'drifter_coriolis_x_' in v]]
    for v in dsmf.variables:
        dsmf[v].attrs = ds_data[v].attrs
    return dsmf
    
def run_cutoff_matchup(l, T=12, cutoff= cutoff):
    ds = cutoff_matchup(l, T=T, cutoff=cutoff).chunk({'obs':500}).persist()
    #store
    zarr = os.path.join(zarr_dir+'_ok','cutoff_matchup',"cutoff_matchup_"+l+"_3.zarr")
    ds.to_zarr(zarr, mode="w")
    logging.info(f"matchup {l} storred in {zarr}")    
    
if __name__ == "__main__":

    ## step0: setup logging

    # to std output
    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    # to file
    logging.basicConfig(
        filename="distributed_cutoff.log",
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

    #overwrite
    if not overwrite :
        labels = [l for l in labels if not os.path.isdir(os.path.join(zarr_dir+'_ok','cutoff_matchup',"cutoff_matchup_"+l+"_3.zarr"))]

    ## boucle for on labels
    #for l in labels: 
    #    if l != 'gps_Sentinel-3_A_2019' : #too big
    #        logging.info(f"start processing {l}")
    #        run_cutoff_matchup(l, T=12, cutoff = cutoff)
    #        logging.info(f"end processing {l}")
       
    l= 'gps_Sentinel-3_A_2019'
    logging.info(f"start processing {l}")
    run_cutoff_matchup(l, T=12, cutoff = cutoff)
    logging.info(f"end processing {l}")

    
    # close dask
    close_dask(cluster, client)

    logging.info("- all done")