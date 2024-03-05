import numpy as np
import pandas as pd
import xarray as xr

def generate_virtual_datacube(extent, period, dl=0.25):
    """Generate a virtual datacube"""
    lon_min, lon_max, lat_min, lat_max = extent
    start, end = period
    latitudes = np.linspace(lat_min, lat_max, int((lat_max-lat_min)/dl))
    longitudes = np.linspace(lon_min, lon_max, int((lon_max-lon_min)/dl))
    times = pd.date_range(start, end, freq='D')
    data = np.random.randn(len(times), len(latitudes), len(longitudes))
    
    return xr.Dataset(
        data_vars=dict(data=(['time', 'lat', 'lon'], data)),
        coords=dict(
            lat=latitudes,
            lon=longitudes,
            time=times,
        )
    )