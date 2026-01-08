# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import zipfile
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from shapely.geometry import Point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import urllib.request
from urllib.error import URLError, HTTPError

# Consideraciones: Hay que revisar el nombre de la columna de trimestral 
# Hay que cambiar dateactual

# ==================== CONFIGURACIÓN ====================
class MapaConfig:
    @staticmethod
    def extraerdir(filezip):
        extract_dir = filezip[:-4]
        os.makedirs(extract_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(filezip, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                print(f"Successfully extracted all files to '{extract_dir}'")
                return extract_dir
        except zipfile.BadZipFile:
            print(f"Error: {filezip} is not a valid zip file.")
            return None
        except FileNotFoundError:
            print(f"Error: {filezip} not found.")
            return None
            
    MESES_ABREV = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    
    MESES_ESPAÑOL = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }

def generar_meses_secuencia(mes_inicio, num_meses=6):
    """Genera secuencia de meses starting from mes_inicio"""
    meses = []
    mes_actual = mes_inicio
    for _ in range(num_meses):
        if mes_actual >= 12:
            mes_actual = 1
        else:
            mes_actual += 1
        meses.append(str(mes_actual))
    return '_'.join(meses)

def descargar_archivo(url, destino):
    """Descarga un archivo desde una URL"""
    try:
        print(f"Descargando: {url}")
        urllib.request.urlretrieve(url, destino)
        print(f" Descarga exitosa: {destino}")
        return True
    except HTTPError as e:
        print(f" Error HTTP {e.code}: {url}")
        return False
    except URLError as e:
        print(f" Error de conexión: {e.reason}")
        return False
    except Exception as e:
        print(f" Error inesperado: {e}")
        return False
    
# ==================== CONFIGURACIÓN INICIAL ====================
# Load shapefile
gdf = gpd.read_file("D:/TEMP ILIANA/Base de datos regional y Monitor de sequía/Monitor Guajira/Fwd Shapefile Colombia/Departamentos_MAGNA.shp")
gdf_filt = gdf[(gdf['DEPARTAMEN'] == 'LA GUAJIRA') | (gdf['DEPARTAMEN'] == 'MAGDALENA') | (gdf['DEPARTAMEN'] == 'CESAR')]

# Separate La Guajira for special styling
gdf_guajira = gdf_filt[gdf_filt['DEPARTAMEN'] == 'LA GUAJIRA']
gdf_other = gdf_filt[gdf_filt['DEPARTAMEN'] != 'LA GUAJIRA']

# Set working directory
wdir = 'D:/TEMP ILIANA/Base de datos regional y Monitor de sequía/Monitor Guajira/03 Mapas de pronostico/'
dateactual = '12-2025'
date_object = datetime.strptime(dateactual, '%m-%Y')

anioactual = dateactual.split('-')[1]
mesactual = int(dateactual.split('-')[0])
mesactualstr = MapaConfig.MESES_ABREV[mesactual]

# Generar secuencia de meses para los archivos zip
meses_secuencia = generar_meses_secuencia(mesactual, 6)

# Calcular próximos meses
datenext = date_object + relativedelta(months=+1)
dateanionext = datenext.year
datemesnext = datenext.month


# URLs base para descarga
base_url = "https://bart.ideam.gov.co/wrfideam/new_modelo/CPT/txt"

# ==================== DESCARGAR ARCHIVOS DE PRECIPITACION ====================

# Nombre del archivo zip de precipitación
filezip_prec = f'{wdir}prec_datos_{mesactualstr}-{anioactual}_{meses_secuencia}.zip'
url_prec = f'{base_url}/PREC/prec_datos_{mesactualstr}-{anioactual}_{meses_secuencia}.zip'

# Descargar archivo de precipitación
if not os.path.exists(filezip_prec):
    descargar_archivo(url_prec, filezip_prec)
else:
    print(f"Archivo ya existe: {filezip_prec}")

# ==================== DESCARGAR ARCHIVOS DE TEMPERATURA ====================

# Descargar datos de temperatura  minima
filezip_tmin = f'{wdir}tmin_datos_{mesactualstr}-{anioactual}_{meses_secuencia}.zip'
url_tmin = f'{base_url}/TMIN/tmin_datos_{mesactualstr}-{anioactual}_{meses_secuencia}.zip'

if not os.path.exists(filezip_tmin):
    descargar_archivo(url_tmin, filezip_tmin)
else:
    print(f"Archivo ya existe: {filezip_tmin}")



# Descargar datos de temperatura  max
filezip_tmax = f'{wdir}tmax_datos_{mesactualstr}-{anioactual}_{meses_secuencia}.zip'
url_tmax = f'{base_url}/TMAX/tmax_datos_{mesactualstr}-{anioactual}_{meses_secuencia}.zip'

if not os.path.exists(filezip_tmax):
    descargar_archivo(url_tmax, filezip_tmax)
else:
    print(f"✓ Archivo ya existe: {filezip_tmax}")
    
    
    
# Descargar datos de temperatura  media
filezip_tmed = f'{wdir}tmed_datos_{mesactualstr}-{anioactual}_{meses_secuencia}.zip'
url_tmed = f'{base_url}/TMED/tmed_datos_{mesactualstr}-{anioactual}_{meses_secuencia}.zip'

if not os.path.exists(filezip_tmed):
    descargar_archivo(url_tmed, filezip_tmed)
else:
    print(f"✓ Archivo ya existe: {filezip_tmed}")

extract_dir_tmed = MapaConfig.extraerdir(filezip_tmed)

if extract_dir_tmed is None:
    raise Exception("No se pudo extraer el archivo de temperatura media")
    

# ==================== CARGAR DATOS DE PRECIPITACIÓN ====================

extract_dir_prec = MapaConfig.extraerdir(filezip_prec)

if extract_dir_prec is None:
    raise Exception("No se pudo extraer el archivo de precipitación")
    
# Cargar datos de precipitación
filepcp01 = f'{extract_dir_prec}/ENSAMBLE_PREC_MENSUAL_{datemesnext:02d}_{dateanionext}.csv'
filepcp02 = f'{extract_dir_prec}/ENSAMBLE_PREC_MENSUAL_{datemesnext+1:02d}_{dateanionext}.csv'
filepcp03 = f'{extract_dir_prec}/ENSAMBLE_PREC_MENSUAL_{datemesnext+2:02d}_{dateanionext}.csv'
filetrimestralpcp = f'{extract_dir_prec}/ENSAMBLE_PREC_TRIMESTRAL_{datemesnext:02d}_{dateanionext}.csv'

dfpcp01 = pd.read_csv(filepcp01, decimal='.', sep=';')[['Longitud','Latitud','Prediccion(mm)','Anomalia(mm)', 'Cond_mas_prob(%)']].copy()
dfpcp02 = pd.read_csv(filepcp02, decimal='.', sep=';')[['Longitud','Latitud','Prediccion(mm)','Anomalia(mm)', 'Cond_mas_prob(%)']].copy()
dfpcp03 = pd.read_csv(filepcp03, decimal='.', sep=';')[['Longitud','Latitud','Prediccion(mm)','Anomalia(mm)', 'Cond_mas_prob(%)']].copy()

dftri = pd.read_csv(filetrimestralpcp, decimal='.', sep=';').rename(columns={'ConProb(%)EFM': 'Cond_mas_prob(%)','Anomalia(mm)EFM': 'Anomalia(mm)'})[['Longitud','Latitud','Anomalia(mm)', 'Cond_mas_prob(%)']].copy()
#dftri = pd.read_csv(filetrimestralpcp, decimal='.', sep=';')[['Longitud','Latitud','Prediccion(mm)','Anomalia(mm)', 'Cond_mas_prob(%)']].copy()

# ==================== CARGAR DATOS DE TEMPERATURA MÍNIMA ====================
filezip_tmin = f'{wdir}tmin_datos_Dec-{anioactual}_1_2_3_4_5_6.zip'
extract_dir_tmin = MapaConfig.extraerdir(filezip_tmin)

filetmin01 = f'{extract_dir_tmin}/TMIN_{datemesnext:02d}_{dateanionext}.csv'
filetmin02 = f'{extract_dir_tmin}/TMIN_{datemesnext+1:02d}_{dateanionext}.csv'
filetmin03 = f'{extract_dir_tmin}/TMIN_{datemesnext+2:02d}_{dateanionext}.csv'

dftmin01 = pd.read_csv(filetmin01, decimal='.', sep=';')[['Longitud','Latitud','Prediccion(°C)','Anomalia(°C)']].copy()
dftmin02 = pd.read_csv(filetmin02, decimal='.', sep=';')[['Longitud','Latitud','Prediccion(°C)','Anomalia(°C)']].copy()
dftmin03 = pd.read_csv(filetmin03, decimal='.', sep=';')[['Longitud','Latitud','Prediccion(°C)','Anomalia(°C)']].copy()

# ==================== CARGAR DATOS DE TEMPERATURA MÁXIMA ====================
filezip_tmax = f'{wdir}tmax_datos_Dec-{anioactual}_1_2_3_4_5_6.zip'
extract_dir_tmax = MapaConfig.extraerdir(filezip_tmax)

filetmax01 = f'{extract_dir_tmax}/TMAX_{datemesnext:02d}_{dateanionext}.csv'
filetmax02 = f'{extract_dir_tmax}/TMAX_{datemesnext+1:02d}_{dateanionext}.csv'
filetmax03 = f'{extract_dir_tmax}/TMAX_{datemesnext+2:02d}_{dateanionext}.csv'

dftmax01 = pd.read_csv(filetmax01, decimal='.', sep=';')[['Longitud','Latitud','Prediccion(°C)','Anomalia(°C)']].copy()
dftmax02 = pd.read_csv(filetmax02, decimal='.', sep=';')[['Longitud','Latitud','Prediccion(°C)','Anomalia(°C)']].copy()
dftmax03 = pd.read_csv(filetmax03, decimal='.', sep=';')[['Longitud','Latitud','Prediccion(°C)','Anomalia(°C)']].copy()

# ==================== CARGAR DATOS DE TEMPERATURA MEDIA ====================
filezip_tmed = f'{wdir}tmed_datos_Dec-{anioactual}_1_2_3_4_5_6.zip'
extract_dir_tmed = MapaConfig.extraerdir(filezip_tmed)

filetmed01 = f'{extract_dir_tmed}/TMED_{datemesnext:02d}_{dateanionext}.csv'
filetmed02 = f'{extract_dir_tmed}/TMED_{datemesnext+1:02d}_{dateanionext}.csv'
filetmed03 = f'{extract_dir_tmed}/TMED_{datemesnext+2:02d}_{dateanionext}.csv'

dftmed01 = pd.read_csv(filetmed01, decimal='.', sep=';')[['Longitud','Latitud','Prediccion(°C)','Anomalia(°C)']].copy()
dftmed02 = pd.read_csv(filetmed02, decimal='.', sep=';')[['Longitud','Latitud','Prediccion(°C)','Anomalia(°C)']].copy()
dftmed03 = pd.read_csv(filetmed03, decimal='.', sep=';')[['Longitud','Latitud','Prediccion(°C)','Anomalia(°C)']].copy()

# ==================== CONFIGURACIÓN DE MAPAS ====================
def get_config_precipitacion():
    colors = ["#f5f1eb","#e6dccf","#d8c7a6","#d1b37a",
              "#e5c94b", "#f2e94e","#c7f000", "#7ee000",
              "#2fd000","#00d28f","#00e5d8","#00cfff",
              "#0094ff","#005bff","#2b3cff","#5a2dff",
              "#7a2cff","#9b2cff","#c42cff","#ff2cff","#ff00ff"]
    levels = [0, 1, 5, 10, 15, 20, 25, 30, 40, 50,
              75, 100, 150, 200, 300, 400, 500,
              600, 800, 1000, 1200]
    return colors, levels

def get_config_anomalia_prec():
    colors_rgb = [(175, 44, 32),(199, 58, 30),(203,73,27),(215,87,22),(227,103,14),
                  (238,118,0),(245,148,0),(248,177,0),(249,206,19),(255,218,98),(255,242,204),
                  (255,255,255),(224,251,241),(192,247,226),(158,243,212),(118,238,198),(0,206,221),
                  (0,184,224),(0,160,223),(0,135,216),(0,108,201),(92,106,238),(173,88,255),(255,0,255)]
    colors = [(r/255, g/255, b/255) for r, g, b in colors_rgb]
    levels = [-400,-350,-300,-250,-200,-150,-75,-50,-25,-10,-7,7,10,25,50,75,100,150,200,250,300,350,400]
    return colors, levels

def get_config_condicion_prob():
    colors_rgb = [(120, 50, 0),(170, 70, 30), (208, 128, 51), (232, 184, 51),(243,243,0),(255,255,255), 
                  (210, 248, 204), (173, 247, 160),(116, 187, 110), (65, 149, 205),(13, 59, 245)]
    colors = [(r/255, g/255, b/255) for r, g, b in colors_rgb]
    levels = [-70,-60,-50,-45,-40,40,45,50,60,70]
    return colors, levels

def get_config_temp_valor():
    #import matplotlib as mpl
    colors =["#4D4CFF","#4D4CFF", "#4CABFF", "#4DFFFF","#BDE0FE","#4E9D4C","#52FF4D","#FFFF4B","#FEC14E","#FF0000","#FF0000"]
    #colors = ["#FF0000","#FF0000","#FEC14E","#FFFF4B","#52FF4D","#4E9D4C","#BDE0FE","#4DFFFF","#4CABFF","#4D4CFF","#4D4CFF"]
    levels = [8,12,16,20,24,28,30,32,34,36]
    #levels = list(range(-6, 40, 1))
    #colors = mpl.colormaps['RdBu'].resampled(47).reversed()
    return colors, levels

def get_config_temp_anomalia():
    #import matplotlib as mpl
    #levels = list(range(-5, 6, 1))
    #colors = mpl.colormaps['RdBu'].resampled(12).reversed()
    colors = ["#0000E6","#0000FF","#0072FD","#01C5FF","#BEE8FF",
              "#E0F7FF",
              "#FFFFFF",
              "#FFF4DB",
              "#FFEAB1","#FFD282","#FFAA01","#FF5500","#E34C00"]
    levels = [-5,-2,-1.5,-1,-0.5,-0.25,0.25,0.5,1,1.5,2,5]
    return colors, levels

# ==================== FUNCIÓN IDW ====================
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

# ==================== FUNCIÓN PRINCIPAL PARA CREAR MAPAS ====================
def crear_mapa(df, var_name, colors, levels, title, output_filename, 
               gdf_filt, gdf_guajira, gdf_other, unit='mm', extend='both'):
    """
    Función para crear y guardar un mapa
    """
    # Convert to Numpy Array
    lon = df['Longitud'].to_numpy()
    lat = df['Latitud'].to_numpy()
    var = df[var_name].to_numpy()
    
    # Prepare known points for IDW
    known_points = np.column_stack((lon, lat))
    known_values = var
    
    # Create target grid
    minx, miny, maxx, maxy = gdf_filt.total_bounds
    grid_space = 0.05
    x_grid = np.arange(minx, maxx, grid_space)
    y_grid = np.arange(miny, maxy, grid_space)
    grid_lon, grid_lat = np.meshgrid(x_grid, y_grid)
    
    # Flatten grid for interpolation
    unknown_points = np.column_stack((grid_lon.flatten(), grid_lat.flatten()))
    
    # Perform IDW interpolation
    idw_result = idw_interpolation(known_points, known_values, unknown_points, power=2, n_neighbors=10)
    grid_var = idw_result.reshape(grid_lon.shape)
    
    # Reproyectar y crear buffer
    gdf_projected = gdf_filt.to_crs('EPSG:21896')
    gdf_buffered = gdf_projected.buffer(5000)
    gdf_buffered = gdf_buffered.to_crs(gdf_filt.crs)
    
    # Mask points outside the shapefile boundary
    mask = np.zeros(grid_lon.shape, dtype=bool)
    for i in range(grid_lon.shape[0]):
        for j in range(grid_lon.shape[1]):
            point = Point(grid_lon[i, j], grid_lat[i, j])
            mask[i, j] = gdf_buffered.contains(point).any()
    
    grid_masked = np.ma.masked_where(~mask, grid_var)
    
    # Create custom colormap
    if isinstance(colors, list):
        custom_cmap = mcolors.ListedColormap(colors)
    else:
        custom_cmap = colors
    
    norm = mcolors.BoundaryNorm(levels, custom_cmap.N, extend=extend)
    
    # Create Figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.2, 1, 0.7], projection=ccrs.PlateCarree())
    
    # Define map boundaries
    extent = [minx - 0.1, maxx + 0.1, miny - 0.1, maxy + 0.1]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add basemap features
    ax.add_feature(cfeature.LAND.with_scale('10m'), edgecolor='black', facecolor='#F8F8F8', zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#D6EAF8', zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='black', zorder=1)
    
    # Add departamentos boundaries
    ax.add_geometries(gdf_other.geometry, ccrs.PlateCarree(), 
                      facecolor='none', edgecolor='black', linewidth=1.5, zorder=3)
    ax.add_geometries(gdf_guajira.geometry, ccrs.PlateCarree(), 
                      facecolor='none', edgecolor='black', linewidth=1.5, zorder=3) #linewidth=3.0, zorder=4)
    
    # Add department names
    for idx, row in gdf_filt.iterrows():
        centroid = row.geometry.centroid
        dept_name = row['DEPARTAMEN']
        ax.text(centroid.x, centroid.y, dept_name, 
                transform=ccrs.PlateCarree(),
                fontsize=9, fontweight='bold', color='black',
                ha='center', va='center', zorder=5)
    
    # Plot interpolated data
    contour = ax.contourf(grid_lon, grid_lat, grid_masked, 
                          levels=levels, cmap=custom_cmap, norm=norm,
                          transform=ccrs.PlateCarree(), extend=extend, zorder=1)
    
    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.6, color='black', alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Add title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Colorbar
    cax = fig.add_axes([0.3, 0.15, 0.6, 0.02])
    cbar = plt.colorbar(contour, cax=cax, orientation='horizontal', 
                        ticks=levels, extend=extend)
    cbar.set_label(f'({unit})', fontsize=12, fontweight='bold')
    if "mm" in var_name:
        cbar.ax.set_xticklabels([f'{v}' for v in levels], rotation=45, ha='center')
    else:
        cbar.ax.set_xticklabels([f'{v}' for v in levels], rotation=0, ha='center')
        
    # North Arrow
    arrow_x, arrow_y = 0.08, 0.88
    ax.annotate('', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y - 0.05),
                arrowprops={'facecolor': 'black', 'width': 5, 'headwidth': 15},
                xycoords=ax.transAxes)
    ax.text(arrow_x, arrow_y + 0.01, 'N', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # Scale Bar
    scale_x_start, scale_y, scale_length = 0.08, 0.12, 0.15
    ax.plot([scale_x_start, scale_x_start + scale_length], [scale_y, scale_y], 
            color='black', linewidth=4, transform=ax.transAxes, zorder=10)
    for i in [0, scale_length]:
        ax.plot([scale_x_start + i, scale_x_start + i], 
                [scale_y - 0.01, scale_y + 0.01],
                color='black', linewidth=3, transform=ax.transAxes, zorder=10)
    ax.text(scale_x_start + scale_length/2, scale_y - 0.025, '50 km',
            ha='center', va='top', fontsize=10, fontweight='bold',
            transform=ax.transAxes, zorder=10)
    
    # Footer info
    fig.text(0.3, 0.08, 'Fuente de datos: Predicción IDEAM', 
             ha='left', va='center', fontsize=13, color='black')
    #fig.text(0.6, 0.91, 'IDW (power=2, n_neighbors=10)', 
    #         ha='left', va='center', fontsize=13, color='grey')
    fig.text(0.45, 0.91, 'Resolución: 0.05°', 
             ha='right', va='center', fontsize=12, style='italic', color='grey')
    
    # Guardar figura
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mapa guardado: {output_filename}")

# ==================== GENERAR TODOS LOS MAPAS ====================
print("Iniciando generación de mapas...")

# Crear directorio de salida si no existe
output_dir = os.path.join(wdir, 'mapas_generados')
os.makedirs(output_dir, exist_ok=True)

# 1. MAPAS DE PRECIPITACIÓN PRONOSTICADA
print("\n1. Generando mapas de precipitación pronosticada...")
colors, levels = get_config_precipitacion()
for i, (df, mes) in enumerate([(dfpcp01, datemesnext), (dfpcp02, datemesnext+1), (dfpcp03, datemesnext+2)], 1):
    mes_nombre = MapaConfig.MESES_ESPAÑOL[mes]
    title = f'Pronóstico de precipitación en La Guajira-Cesar-Magdalena\n{mes_nombre} {dateanionext}'
    filename = f'{output_dir}/01_prec_pronostico_mes{i}_{mes_nombre}_{dateanionext}.png'
    crear_mapa(df, 'Prediccion(mm)', colors, levels, title, filename, 
               gdf_filt, gdf_guajira, gdf_other, unit='mm', extend='max')

# Trimestral
#title = f'Pronóstico de Precipitación Trimestral\nLa Guajira - {dateanionext}'
#filename = f'{output_dir}/01_prec_pronostico_trimestral_{dateanionext}.png'
#crear_mapa(dftri, 'Prediccion(mm)', colors, levels, title, filename, 
#           gdf_filt, gdf_guajira, gdf_other, unit='mm', extend='max')

# 2. MAPAS DE ANOMALÍA DE PRECIPITACIÓN
print("\n2. Generando mapas de anomalía de precipitación...")
colors, levels = get_config_anomalia_prec()
for i, (df, mes) in enumerate([(dfpcp01, datemesnext), (dfpcp02, datemesnext+1), (dfpcp03, datemesnext+2)], 1):
    mes_nombre = MapaConfig.MESES_ESPAÑOL[mes]
    title = f'Anomalía de precipitación en La Guajira-Cesar-Magdalena\n{mes_nombre} {dateanionext}'
    filename = f'{output_dir}/02_prec_anomalia_mes{i}_{mes_nombre}_{dateanionext}.png'
    crear_mapa(df, 'Anomalia(mm)', colors, levels, title, filename, 
               gdf_filt, gdf_guajira, gdf_other, unit='mm', extend='both')

# Trimestral
title = f'Anomalía de precipitación trimestral en La Guajira-Cesar-Magdalena\n{dateanionext}'
filename = f'{output_dir}/02_prec_anomalia_trimestral_{dateanionext}.png'
crear_mapa(dftri, 'Anomalia(mm)', colors, levels, title, filename, 
           gdf_filt, gdf_guajira, gdf_other, unit='mm', extend='both')

# 3. MAPAS DE CONDICIÓN MÁS PROBABLE
print("\n3. Generando mapas de condición más probable...")
colors, levels = get_config_condicion_prob()
for i, (df, mes) in enumerate([(dfpcp01, datemesnext), (dfpcp02, datemesnext+1), (dfpcp03, datemesnext+2)], 1):
    mes_nombre = MapaConfig.MESES_ESPAÑOL[mes]
    title = f'Condición más probable de precipitación en La Guajira-Cesar-Magdalena\n{mes_nombre} {dateanionext}'
    filename = f'{output_dir}/03_cond_mas_prob_mes{i}_{mes_nombre}_{dateanionext}.png'
    crear_mapa(df, 'Cond_mas_prob(%)', colors, levels, title, filename, 
               gdf_filt, gdf_guajira, gdf_other, unit='%', extend='both')

# Trimestral
title = f'Condición más probable trimestral en La Guajira-Cesar-Magdalena\n{dateanionext}'
filename = f'{output_dir}/03_cond_mas_prob_trimestral_{dateanionext}.png'
crear_mapa(dftri, 'Cond_mas_prob(%)', colors, levels, title, filename, 
           gdf_filt, gdf_guajira, gdf_other, unit='%', extend='both')

# 4. MAPAS DE TEMPERATURA MÍNIMA PRONOSTICADA
print("\n4. Generando mapas de temperatura mínima pronosticada...")
colors, levels = get_config_temp_valor()
for i, (df, mes) in enumerate([(dftmin01, datemesnext), (dftmin02, datemesnext+1), (dftmin03, datemesnext+2)], 1):
    mes_nombre = MapaConfig.MESES_ESPAÑOL[mes]
    title = f'Pronóstico de temperatura mínima en La Guajira-Cesar-Magdalena\n{mes_nombre} {dateanionext}'
    filename = f'{output_dir}/04_tmin_pronostico_mes{i}_{mes_nombre}_{dateanionext}.png'
    crear_mapa(df, 'Prediccion(°C)', colors, levels, title, filename, 
               gdf_filt, gdf_guajira, gdf_other, unit='°C', extend='both')

# 5. MAPAS DE ANOMALÍA DE TEMPERATURA MÍNIMA
print("\n5. Generando mapas de anomalía de temperatura mínima...")
colors, levels = get_config_temp_anomalia()
for i, (df, mes) in enumerate([(dftmin01, datemesnext), (dftmin02, datemesnext+1), (dftmin03, datemesnext+2)], 1):
    mes_nombre = MapaConfig.MESES_ESPAÑOL[mes]
    title = f'Anomalía de temperatura mínima en La Guajira-Cesar-Magdalena\n{mes_nombre} {dateanionext}'
    filename = f'{output_dir}/05_tmin_anomalia_mes{i}_{mes_nombre}_{dateanionext}.png'
    crear_mapa(df, 'Anomalia(°C)', colors, levels, title, filename, 
               gdf_filt, gdf_guajira, gdf_other, unit='°C', extend='both')

# 6. MAPAS DE TEMPERATURA MÁXIMA PRONOSTICADA
print("\n6. Generando mapas de temperatura máxima pronosticada...")
colors, levels = get_config_temp_valor()
for i, (df, mes) in enumerate([(dftmax01, datemesnext), (dftmax02, datemesnext+1), (dftmax03, datemesnext+2)], 1):
    mes_nombre = MapaConfig.MESES_ESPAÑOL[mes]
    title = f'Pronóstico de Temperatura Máxima en La Guajira-Cesar-Magdalena\n{mes_nombre} {dateanionext}'
    filename = f'{output_dir}/06_tmax_pronostico_mes{i}_{mes_nombre}_{dateanionext}.png'
    crear_mapa(df, 'Prediccion(°C)', colors, levels, title, filename, 
               gdf_filt, gdf_guajira, gdf_other, unit='°C', extend='both')

# 7. MAPAS DE ANOMALÍA DE TEMPERATURA MÁXIMA
print("\n7. Generando mapas de anomalía de temperatura máxima...")
colors, levels = get_config_temp_anomalia()
for i, (df, mes) in enumerate([(dftmax01, datemesnext), (dftmax02, datemesnext+1), (dftmax03, datemesnext+2)], 1):
    mes_nombre = MapaConfig.MESES_ESPAÑOL[mes]
    title = f'Anomalía de temperatura máxima en La Guajira-Cesar-Magdalena\n{mes_nombre} {dateanionext}'
    filename = f'{output_dir}/07_tmax_anomalia_mes{i}_{mes_nombre}_{dateanionext}.png'
    crear_mapa(df, 'Anomalia(°C)', colors, levels, title, filename, 
               gdf_filt, gdf_guajira, gdf_other, unit='°C', extend='both')

# 8. MAPAS DE TEMPERATURA MEDIA PRONOSTICADA
print("\n8. Generando mapas de temperatura media pronosticada...")
colors, levels = get_config_temp_valor()
for i, (df, mes) in enumerate([(dftmed01, datemesnext), (dftmed02, datemesnext+1), (dftmed03, datemesnext+2)], 1):
    mes_nombre = MapaConfig.MESES_ESPAÑOL[mes]
    title = f'Pronóstico de Temperatura Media en La Guajira-Cesar-Magdalena\n{mes_nombre} {dateanionext}'
    filename = f'{output_dir}/08_tmed_pronostico_mes{i}_{mes_nombre}_{dateanionext}.png'
    crear_mapa(df, 'Prediccion(°C)', colors, levels, title, filename, 
               gdf_filt, gdf_guajira, gdf_other, unit='°C', extend='both')

# 9. MAPAS DE ANOMALÍA DE TEMPERATURA MEDIA
print("\n9. Generando mapas de anomalía de temperatura media...")
colors, levels = get_config_temp_anomalia()
for i, (df, mes) in enumerate([(dftmed01, datemesnext), (dftmed02, datemesnext+1), (dftmed03, datemesnext+2)], 1):
    mes_nombre = MapaConfig.MESES_ESPAÑOL[mes]
    title = f'Anomalía de temperatura media en La Guajira-Cesar-Magdalena\n{mes_nombre} {dateanionext}'
    filename = f'{output_dir}/09_tmed_anomalia_mes{i}_{mes_nombre}_{dateanionext}.png'
    crear_mapa(df, 'Anomalia(°C)', colors, levels, title, filename, 
               gdf_filt, gdf_guajira, gdf_other, unit='°C', extend='both')
