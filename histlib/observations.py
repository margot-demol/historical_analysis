import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import os
from pathlib import Path
from datetime import datetime, timedelta

path = Path(os.path.dirname(__file__))
gdf_ocean = gpd.read_file(path / "_data/ne_50m_ocean.shp")
gdf_ocean = gdf_ocean.rename(index={0: 'ocean'})

def generate_ocean_coordinates(N, extent=(-180, 180, -90, 90)):
    """Generate N coordinates inside oceans"""
    lon_min, lon_max, lat_min, lat_max = extent
    gdf_points_N = gpd.GeoDataFrame()
    n=1
    while len(gdf_points_N) < N:
        #print(n)
        lon = np.random.uniform( lon_min, lon_max, N)
        lat = np.random.uniform( lat_min, lat_max, N)
        df = pd.DataFrame()
        df['points'] = list(zip(lon,lat))
        df['points'] = df['points'].apply(Point)
        gdf_points = gpd.GeoDataFrame(df, geometry='points')
        gdf_points = gdf_points.set_crs('EPSG:4326')
        Sjoin = gpd.tools.sjoin(gdf_points, gdf_ocean, predicate="within", how='left').dropna()
        gdf_points_N = pd.concat([gdf_points_N, Sjoin], ignore_index=True)
        #print(f"   {len(gdf_points_N)}")
        n+=1
    return gdf_points_N[['points']].head(N)

def generate_ocean_observations(N, extent, period):
    """Generate N observations inside oceans"""
    obs = np.arange(0, N)
    start, end = period
    start_dt, end_dt = datetime.strptime(start, "%Y-%m-%d"), datetime.strptime(end, "%Y-%m-%d")
    time = np.array([start_dt + timedelta(days=np.random.randint(0, int((end_dt - start_dt).days))) for obs in range(N)])
    gdf = generate_ocean_coordinates(N, extent=extent)
    longitude, latitude = gdf.points.x.to_numpy(), gdf.points.y.to_numpy()
    return time, longitude, latitude, obs

def generate_virtual_observations(N, extent, period):
    """Generate a xarray.Dataset of N virtual observations inside oceans"""
    times, longitudes, latitudes, obs = generate_ocean_observations(N, extent, period)
    return xr.Dataset(
        coords=dict(
            obs=obs,
            lat=xr.DataArray(latitudes,  dims=["obs"]),
            lon=xr.DataArray(longitudes,  dims=["obs"]),
            time=xr.DataArray(times,  dims=["obs"]),
        )
    )

def plot_observations(ds):
    """Plot a map with observations point"""
    # Carte des observations
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.02)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.scatter(ds.lon, ds.lat, color='red', marker='o', s=10, transform=ccrs.PlateCarree())#, ax=ax)

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.OCEAN, color='lightblue')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    ax.set_global()

    plt.title(f'Coordonnées des observations', fontsize=20) 