"""
Ce code cherche à comparer le temps d'éxecution des 2 méthodes (sequentiel & flatten vectorized) de selection de données au sein d'un datacube. 
"""
import sys
import uuid
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

import utils
from methods import benchmark

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Datacube Select Points',
        description='Program to optimise the selection of points in a datacube'
    )
    # METHODS
    parser.add_argument('--method', type=str, default=['sequential'], nargs='+', help='list of selection methods')
    parser.add_argument('--dt', type=int, default=[2], nargs='+', help='list of time radius of the box (+/- dt days)') 
    parser.add_argument('--dlon', type=int, default=[5], nargs='+', help='list of longitude radius of the box (+/- dlon° of longitude)')
    parser.add_argument('--dlat', type=int, default=[5], nargs='+', help='list of latitude radius of the box (+/- dlat° of latitude)')
    parser.add_argument('--dl', type=int, default=[0.25], nargs='+', help='list of lon/lat step between values')
    # OBSERVATION
    parser.add_argument('--number-obs', type=int, default=[100], nargs='+', help='list of number of observation points')
    parser.add_argument('--extent', type=int, default=[-180,180,-90,90], nargs=4, help="geographic extent of observation points [lon_min, lon_max, lat_min, lat_max]")
    parser.add_argument('--period', type=str, default=['2000-01-01', '2000-01-02'], nargs=2, help="time period of observation points [start, end] (format: '2000-01-01')")
    # DATACUBE
    parser.add_argument('--time-chunks', type=int, default=[5], nargs='+', help='list of time chunk sizes')
    parser.add_argument('--geo-chunks', type=int, default=[150], nargs='+', help='list of space chunk sizes')
    parser.add_argument('--datacube-size', type=int, default=[1], nargs='+', help='datacube size list')
    # CLUSTER
    parser.add_argument('--queue', type=str, default=['omp'], nargs='+', help='PBS queue list')
    parser.add_argument('--jobs', type=int, default=[2], nargs='+', help='list of the number of PBS jobs')
    parser.add_argument('--cores', type=int, default=[4], nargs='+', help='list of the number of PBS cores per job')
    parser.add_argument('--memory', type=str, default=['8G'], nargs='+', help='list of the number of PBS memory per job')
    parser.add_argument('--processes', type=int, default=[1], nargs='+', help='list of the number of PBS processes per job')
    parser.add_argument('--interface', type=str, default='ib0', help='communication interface for PBS jobs')
    parser.add_argument('--walltime', type=str, default='01:00:00', help='walltime for PBS jobs' )
    # OTHER
    parser.add_argument('--output-dir', type=Path, required=True, help='path for output files')
    args = parser.parse_args()

    # PATH
    utils.create_dir(args.output_dir)

    # LOGGING
    result = utils.create_logger('result', args.output_dir / f'result.log', format="%(message)s", mode='a')                 
    for method in args.method:
        for dt in args.dt:
            for dlon in args.dlon:
                for dlat in args.dlat:
                    for dl in args.dl:
                        for number_obs in args.number_obs:
                            for time_chunk in args.time_chunks:
                                for geo_chunk in args.geo_chunks:
                                    for datacube_size in args.datacube_size:
                                        for queue in args.queue:
                                            for jobs in args.jobs:
                                                for cores in args.cores:
                                                    for memory in args.memory:
                                                        for processes in args.processes:
                                                            
                                                            str_uuid = str(uuid.uuid4())
                                                            
                                                            run_time_total = benchmark(method, dt, dlon, dlat, dl,
                                                                                        number_obs, args.extent, args.period,
                                                                                        time_chunk, geo_chunk, datacube_size, 
                                                                                        queue, jobs, cores, memory, processes, args.interface, args.walltime,
                                                                                        args.output_dir, str_uuid)

                                                            extent = tuple(args.extent)
                                                            start, end = tuple(args.period)
                                                            start_dt, end_dt = datetime.strptime(start, "%Y-%m-%d"), datetime.strptime(end, "%Y-%m-%d")
                                                            interval = (end_dt - start_dt).days
                                                            
                                                            # RESULT
                                                            result_dict = dict(
                                                                id=str_uuid,
                                                                # METHODS
                                                                method=method,
                                                                # OBSERVATIONS
                                                                number_obs=number_obs,
                                                                extent=extent,
                                                                start=start,
                                                                end=end,
                                                                interval=interval,
                                                                # DATACUBE
                                                                time_chunk=time_chunk,
                                                                geo_chunk=geo_chunk,
                                                                datacube_size=datacube_size,
                                                                # CLUSTER
                                                                queue=queue,
                                                                jobs=jobs,
                                                                cores=cores,
                                                                memory=memory,
                                                                processes=processes,
                                                                interface=args.interface,
                                                                walltime=args.walltime,
                                                                # RESULT
                                                                run_time_total=run_time_total,
                                                            )

                                                            result.info(result_dict)