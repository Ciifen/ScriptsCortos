# -*- coding: utf-8 -*-

# Import libraries
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
import matplotlib.ticker as mticker
from scipy.spatial import distance
from shapely.geometry import Point

# Set working directory
wdir = 'D:/TEMP ILIANA/Base de datos regional y Monitor de sequía/Monitor Guajira/Mapa de precipitacion mensual/'

# Load and prepare data
df = pd.read_excel("D:/TEMP ILIANA/Base de datos regional y Monitor de sequía/Monitor Guajira/Mapa de precipitacion mensual/PREC-OCTUBRE_25.xlsx", 
                   sheet_name="BANCO DE DATOS", header=1)
df = df.dropna(subset=['DPTO'])
df['DPTO'] = df['DPTO'].str.upper()
df = df[df['DPTO'].isin(['CESAR','LA GUAJIRA','MAGDALENA'])]
df = df.dropna(subset=['PREC\nTOTAL'])
df = df[['lon','lat','PREC\nTOTAL']]

# Convert to Numpy Array
lon = df['lon'].to_numpy()
lat = df['lat'].to_numpy()
rainfall = df['PREC\nTOTAL'].to_numpy()

# Load shapefile
gdf = gpd.read_file("D:/TEMP ILIANA/Base de datos regional y Monitor de sequía/Monitor Guajira/Fwd Shapefile Colombia/Departamentos_MAGNA.shp")
gdf_filt = gdf[(gdf['DEPARTAMEN'] == 'LA GUAJIRA') | (gdf['DEPARTAMEN'] == 'MAGDALENA') | (gdf['DEPARTAMEN'] == 'CESAR')]

# Separate La Guajira for special styling
gdf_guajira = gdf_filt[gdf_filt['DEPARTAMEN'] == 'LA GUAJIRA']
gdf_other = gdf_filt[gdf_filt['DEPARTAMEN'] != 'LA GUAJIRA']

# =====================================================
# IDW INTERPOLATION
# =====================================================

# Prepare known points for IDW
known_points = np.column_stack((lon, lat))
known_values = rainfall

# Create target grid
minx, miny, maxx, maxy = gdf_filt.total_bounds
grid_space = 0.05
x_grid = np.arange(minx, maxx, grid_space)
y_grid = np.arange(miny, maxy, grid_space)
grid_lon, grid_lat = np.meshgrid(x_grid, y_grid)

# Flatten grid for interpolation
unknown_points = np.column_stack((grid_lon.flatten(), grid_lat.flatten()))

def idw_interpolation(known_points, known_values, unknown_points, power=2, n_neighbors=10):
    """Inverse Distance Weighting interpolation"""
    interpolated = np.zeros(len(unknown_points))
    
    for i, unknown_pt in enumerate(unknown_points):
        distances = np.sqrt(np.sum((known_points - unknown_pt)**2, axis=1))
        
        if np.min(distances) < 1e-10:
            interpolated[i] = known_values[np.argmin(distances)]
            continue
        
        if n_neighbors < len(distances):
            nearest_indices = np.argpartition(distances, n_neighbors)[:n_neighbors]
            distances = distances[nearest_indices]
            values = known_values[nearest_indices]
        else:
            values = known_values
        
        weights = 1 / (distances ** power)
        weights = weights/np.sum(weights)
        interpolated[i] = np.sum(weights * values)
    
    return interpolated

idw_result = idw_interpolation(known_points, known_values, unknown_points, power=2, n_neighbors=10)

# Reshape interpolated values to grid shape
grid_rainfall = idw_result.reshape(grid_lon.shape)

# Mask points outside the shapefile boundary
mask = np.zeros(grid_lon.shape, dtype=bool)
for i in range(grid_lon.shape[0]):
    for j in range(grid_lon.shape[1]):
        point = Point(grid_lon[i, j], grid_lat[i, j])
        mask[i, j] = gdf_filt.contains(point).any()

grid_rainfall_masked = np.ma.masked_where(~mask, grid_rainfall)

# =====================================================
# VISUALIZATION
# =====================================================

# Create custom colormap
levels = [0, 50, 100, 150, 200, 300, 400, 600, 800, 1000]
color = ['#FD0000','#F9AB03','#FEFC02','#A2FF6D','#4CE600',
         '#39A602','#72DEFE','#00C6FE','#0071FC', '#E16FFF']
custom_cmap = mcolors.ListedColormap(color)
norm = mcolors.BoundaryNorm(levels, custom_cmap.N, extend='max')

# Map Title
title = 'Precipitación en La Guajira \n Octubre 2025'

# Create Figure with adjusted spacing for bottom text
fig = plt.figure(figsize=(10, 10))

# [left, bottom, width, height]
ax = fig.add_axes([0.1, 0.2, 1, 0.7], projection=ccrs.PlateCarree())

# Define map boundaries
extent = [minx - 0.1, maxx + 0.1, miny - 0.1, maxy + 0.1]
ax.set_extent(extent, crs=ccrs.PlateCarree())

# Add basemap features
ax.add_feature(cfeature.LAND.with_scale('10m'), edgecolor='black', facecolor='#F8F8F8', zorder=0)
ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#D6EAF8', zorder=0)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='black', zorder=1)

# Add departamentos boundaries - other departments (thinner)
ax.add_geometries(gdf_other.geometry, ccrs.PlateCarree(), 
                  facecolor='none', edgecolor='black', linewidth=1.5, zorder=3)

# Add La Guajira boundary (thicker to highlight)
ax.add_geometries(gdf_guajira.geometry, ccrs.PlateCarree(), 
                  facecolor='none', edgecolor='black', linewidth=3.0, zorder=4)

# Add department names
for idx, row in gdf_filt.iterrows():
    centroid = row.geometry.centroid
    dept_name = row['DEPARTAMEN']
    ax.text(centroid.x, centroid.y, dept_name, 
            transform=ccrs.PlateCarree(),
            fontsize=9, fontweight='bold', color='dimgrey',
            ha='center', va='center', zorder=5)

# Plot interpolated rainfall using contourf
contour = ax.contourf(grid_lon, grid_lat, grid_rainfall_masked, 
                      levels=levels, cmap=custom_cmap, norm=norm,
                      transform=ccrs.PlateCarree(), extend='max', zorder=1)

# Plot station points
#stations = ax.scatter(lon, lat, c=rainfall, norm=norm, cmap=custom_cmap, 
#                     edgecolors='black', linewidths=1.5, 
#                     transform=ccrs.PlateCarree(), s=50,
#                     label='Estaciones', zorder=6)

# Add gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.6, color='black', alpha=0.3, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Add title
ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

# =====================================================
# COLORBAR
# =====================================================
# Create colorbar axis manually for better control
# [left, bottom, width, height]
cax = fig.add_axes([0.3, 0.15, 0.6, 0.02])
cbar = plt.colorbar(contour, cax=cax, orientation='horizontal', 
                    ticks=levels, extend='max')
cbar.set_label('Precipitación (mm/mes)', fontsize=12, fontweight='bold')
cbar.ax.set_xticklabels(['0', '50', '100', '150', '200', '300', '400', '600', '800', '>1000'])

# =====================================================
# NORTH ARROW
# =====================================================
arrow_x = 0.08
arrow_y = 0.88

ax.annotate('',
            xy=(arrow_x, arrow_y),          # Punta de la flecha
            xytext=(arrow_x, arrow_y - 0.05),# Base de la flecha
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            xycoords=ax.transAxes)

# Letra N
ax.text(arrow_x, arrow_y + 0.01, 'N',
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=16, fontweight='bold')

# =====================================================
# SCALE BAR
# =====================================================
scale_x_start = 0.08
scale_y = 0.12
scale_length = 0.15

ax.plot([scale_x_start, scale_x_start + scale_length], 
        [scale_y, scale_y], 
        color='black', linewidth=4, transform=ax.transAxes, zorder=10)

for i in [0, scale_length]:
    ax.plot([scale_x_start + i, scale_x_start + i], 
            [scale_y - 0.01, scale_y + 0.01],
            color='black', linewidth=3, transform=ax.transAxes, zorder=10)

ax.text(scale_x_start + scale_length/2, scale_y - 0.025, 
        '50 km',
        ha='center', va='top', fontsize=10, fontweight='bold',
        transform=ax.transAxes, zorder=10)

# =====================================================
# LEGEND
# =====================================================
#ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

# =====================================================
# Informacion figura lineas alrededor grafico
# =====================================================


fig.text(0.3, 0.08, 'Fuente de datos: IDEAM', 
         ha='left', va='center', fontsize=13,
         color='black')


fig.text(0.6, 0.91, 'IDW (power=2, n_neighbors=10)', 
         ha='left', va='center', fontsize=13, 
         color='grey')


fig.text(0.45, 0.91, 'Resolución: 0.05°', 
         ha='right', va='center', fontsize=12, style='italic', 
         color='grey')

# Save figure
plt.savefig(wdir + 'Precipitation_IDW_Map_Octubre2025.png',
            bbox_inches='tight', dpi=300)
plt.show()

print(f"Mapa guardado exitosamente en: {wdir}")
print(f"Número de estaciones utilizadas: {len(lon)}")
print(f"Rango de precipitación: {rainfall.min():.1f} - {rainfall.max():.1f} mm")