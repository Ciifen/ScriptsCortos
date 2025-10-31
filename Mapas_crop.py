#!/usr/bin/env python
import os
import sys
import rasterio
import numpy as np
import pandas as pd
import datetime as dt
import geopandas as gpd
from rasterio.mask import mask
from scipy.interpolate import griddata
from rasterio.transform import from_origin
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib as mpl
import tempfile
from rasterio.enums import Resampling
from rasterio.transform import Affine
from shapely.geometry import box

#Ejecutar env con source/var/py/indices-monitor/env-indices/bin/activate
# Ejemplo: python3.9 Mapas_crop.py "01" "2024_02" "sdi" "-70" "-61" "-20" "-15" "/var/py/castehr"

#lapso = sys.argv[1]
#fecha = sys.argv[2]
#indice = sys.argv[3]
#xmin = sys.argv[4]; xmax =sys.argv[5]; ymin=sys.argv[6]; ymax=sys.argv[7]
#outSave = sys.argv[8]

lapso = '12' #01,02,03,04
fecha = "2024_12" #yyyy_mm
indice = "spi" # mon, sdi, sndvi, spei, spi, ssmi,
outSave = "/var/py/castehr/data/indices" # Donde almacenar las salidas
xmin = -81; xmax = -78; ymin=-3; ymax=0

# Nombre de archivo de entrada
nameTif = f"/var/py/castehr/data/indices/{indice}/tif/{fecha}_{lapso}.tif" 

# Nombres de archivos de salida del png
filepng = outSave + os.path.basename(nameTif).replace(".tif", ".png")
fileTif_resampled = outSave + os.path.basename(nameTif).replace(".tif", ".png")
fileTif_cropped = outSave + os.path.basename(nameTif).replace(".tif", ".png")
 

def resample_raster(input_path, output_path, upscale_factor=25, resampling_method=Resampling.bilinear):
    """
    Remuestrea un raster aumentando su resolución.
    
    Parameters:
     - input_path (str): Ruta del raster de entrada
     - output_path (str): Ruta donde guardar el raster remuestreado
     - upscale_factor (int): Factor de aumento de resolución
     - resampling_method: Método de remuestreo (bilinear, cubic, etc.)
    
    Returns:
     - str: Ruta del archivo remuestreado
    """
    with rasterio.open(input_path) as src:
        # Calcular nuevas dimensiones
        new_width = int(src.width * upscale_factor)
        new_height = int(src.height * upscale_factor)
        
        # Calcular nueva transformación
        new_transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )
        
        # Remuestrear datos
        resampled_data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=resampling_method
        )
        
        # Actualizar perfil
        profile = src.profile.copy()
        profile.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform,
            'dtype': resampled_data.dtype
        })
        
        # Guardar raster remuestreado
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(resampled_data)
    
    print(f"Raster remuestreado guardado en: {output_path}")
    return output_path


def crop_raster_by_bbox(input_path, output_path, xmin, xmax, ymin, ymax):
    """
    Recorta un raster usando coordenadas de bounding box y guarda el resultado.
    
    Parameters:
     - input_path (str): Ruta del raster de entrada
     - output_path (str): Ruta donde guardar el raster recortado
     - xmin, xmax, ymin, ymax: Coordenadas del área de recorte
    
    Returns:
     - tuple: (datos recortados, transformación)
    """
    with rasterio.open(input_path) as src:
        # Crear geometría del bounding box
        bbox_geom = box(xmin, ymin, xmax, ymax)
        geo = gpd.GeoDataFrame({'geometry': [bbox_geom]}, crs=src.crs)
        
        # Realizar el recorte
        cropped_data, cropped_transform = mask(src, geo.geometry, crop=True)
        
        # Actualizar metadatos
        cropped_meta = src.meta.copy()
        cropped_meta.update({
            "driver": "GTiff",
            "height": cropped_data.shape[1],
            "width": cropped_data.shape[2],
            "transform": cropped_transform
        })
        
        # Guardar raster recortado
        with rasterio.open(output_path, "w", **cropped_meta) as dest:
            dest.write(cropped_data)
    
    print(f"Raster recortado guardado en: {output_path}")
    return cropped_data, cropped_transform


def plot_raster(raster_url: str, gdf: gpd.GeoDataFrame, fig_name: str, xmin, mmax, ymin, ymax, lapso, indice, fecha, suavizar=True, upscale_factor=25, 
                fileTif_resampled=None, fileTif_cropped=None) -> None:
    """
    Genera un mapa a partir de un raster y lo guarda como imagen PNG.
    
    Parameters:
     - raster_url (str): Path to the input raster file.
     - gdf (GeoDataFrame): GeoDataFrame containing the geometries for masking.
     - fig_name (str): Output figure file name.
     - xmin, xmax, ymin, ymax: Coordenadas del área de visualización
     - lapso (str): Número de meses
     - indice (str): Tipo de índice (spi, spei, etc.)
     - fecha (str): Fecha en formato yyyy_mm
     - suavizar (bool): Si se aplica remuestreo para suavizar
     - upscale_factor (int): Factor de suavizado
     - fileTif_resampled (str): Ruta para guardar el raster remuestreado (opcional)
     - fileTif_cropped (str): Ruta para guardar el raster recortado (opcional)
    """
    try:
        # Generar rutas automáticamente si no se proporcionan
        if fileTif_resampled is None:
            fileTif_resampled = fig_name.replace(".png", "_resampled.tif")
        if fileTif_cropped is None:
            fileTif_cropped = fig_name.replace(".png", "_cropped.tif")
            
        dictIndices = {'mon':"Monitor de sequía de Ecuador a:", 
                    'sdi':"Índice hidrológico de sequía de Nalbantis a:",
                    'sndvi':"Índice estandarizado de vegetación (SNDVI) a:",
                    'spei':"Índice estandarizado de evapotranspiración a:",
                    'spi':"Índice estandarizado de precipitación (SPI) a:",
                    'ssmi':"Índice estandarizado de humedad del suelo (SSMI) a:",
                    'sti':"Índice estandarizado de temperatura a:"}
        # Variable para almacenar la ruta del raster a usar
        raster_to_use = raster_url
        
        # Aplicar suavizado si está activado
        if suavizar:
            print("Aplicando remuestreo para suavizado...")
            raster_to_use = resample_raster(raster_url, fileTif_resampled, upscale_factor)
        
        # Realizar el recorte por coordenadas y guardarlo
        print("Recortando raster por coordenadas...")
        cropped_data, cropped_transform = crop_raster_by_bbox(
            raster_to_use, fileTif_cropped, xmin, xmax, ymin, ymax
        )
            
        
        # Leer el raster recortado y aplicar máscara de geometría
        with rasterio.open(fileTif_cropped) as src:
            # Leer los datos del raster
            out_image_masked, out_transform = mask(src, gdf.geometry, crop=True)
 
        #Convertir a un arreglo de float64
        out_image_masked = out_image_masked.astype(np.float64)
            
    
        if indice=="mon":
            # Reemplazar valores menores a 0
            out_image_masked = np.where(out_image_masked < 0, np.nan, out_image_masked)
        else:
            # Reemplazar valores menores a -20 o mayores a 20 con NaN
            out_image_masked = np.where((out_image_masked < -20) | (out_image_masked > 20), np.nan, out_image_masked)

        # Asegurarse de que no haya valores NaN
        out_image_masked = np.nan_to_num(out_image_masked, nan=np.nan)

        # Configurar colores
        if indice=="mon":
            # Crear una lista de colores utilizando la función color
            lstColors = ['#cccccc','#fffe00','#fdd37f','#ffaa00','#fe0002','#710100']
            cmap_custom = ListedColormap(lstColors)
        
            # Definir los límites de los intervalos
            bounds = [0, 1, 2, 3, 4, 5, 6]  # Notar que hay un límite adicional para cubrir todo el rango
            norm = BoundaryNorm(bounds, ncolors=len(lstColors), extend='neither')
            
        else:
            def color(pixelValue: float) -> str:
                if -10.0 <= pixelValue < -2.5:
                    return '#890002'
                elif -2.5 <= pixelValue < -2.0:
                    return '#C50000'
                elif -2.0 <= pixelValue < -1.5:
                    return '#F50302'
                elif -1.5 <= pixelValue < -1.0:
                    return '#F08125'
                elif -1.0 <= pixelValue < -0.5:
                    return '#E8AF2E'
                elif -0.5 <= pixelValue < 0.5:
                    return '#D3D3D3'  # Gris para el rango de -0.25 a 0.25
                elif 0.5 <= pixelValue < 1.0:
                    return '#00A904'
                elif 1.0 <= pixelValue < 1.5:
                    return '#00A0FF'
                elif 1.5 <= pixelValue < 2.0:
                    return '#8000E1'
                elif 2.0 <= pixelValue < 2.5:
                    return '#A101C7'
                elif 2.5 <= pixelValue <= 10.0:
                    return '#E50FED'
                else:
                   # Valor fuera del rango: asigna un color neutro o transparente
                   return '#FFFFFF'  # o '#00000000' si quieres transparente


            # Definir valores mínimos y máximos
            mmin = -3; mmax = 3

            # Crear una lista de valores entre 0 y 1
            values = np.linspace(mmin, mmax, 500)  # Limitar a un máximo de 1000 colores
        
            # Crear una lista de colores utilizando la función color
            colors = [color(value) for value in values]
            cmap_custom = ListedColormap(colors)

            
        # Crea una figura de Matplotlib y muestra el raster enmascarado
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.margins(0)
        
        if indice=="mon":
            # Mostrar la imagen del raster usando imshow para obtener el mapeador
            img = ax.imshow(out_image_masked[0], cmap=cmap_custom, norm=norm, extent=(
                out_transform[2], 
                out_transform[2] + out_transform[0] * out_image_masked.shape[2],
                out_transform[5] + out_transform[4] * out_image_masked.shape[1], 
                out_transform[5]
            ))

        else:
            # Mostrar la imagen del raster usando imshow para obtener el mapeador
            img = ax.imshow(out_image_masked[0], cmap=cmap_custom, extent=(
                out_transform[2], 
                out_transform[2] + out_transform[0] * out_image_masked.shape[2],
                out_transform[5] + out_transform[4] * out_image_masked.shape[1], 
                out_transform[5]
            ), vmin=-3, vmax=3)

        gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5)

        # Establecer límites en los ejes x e y
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        # Agregar leyenda/barra de color
        if indice=="mon":
            # Agregar la barra de color
            lstIntervalsLbls = ['Sequía excepcional','Sequía extrema','Sequía severa','Sequía moderada','Anormalmente seco','Normal']
            custom_lines = [
                    mpl.lines.Line2D([0], [0], color=lstColors[5], lw=5),
                    mpl.lines.Line2D([0], [0], color=lstColors[4], lw=5),
                    mpl.lines.Line2D([0], [0], color=lstColors[3], lw=5),
                    mpl.lines.Line2D([0], [0], color=lstColors[2], lw=5),
                    mpl.lines.Line2D([0], [0], color=lstColors[1], lw=5),
                    mpl.lines.Line2D([0], [0], color=lstColors[0], lw=5)]
            plt.legend(bbox_to_anchor=(1,1), loc="upper left",handles=custom_lines, labels=lstIntervalsLbls,title="   Intensidad de sequía   ",title_fontsize='large',fontsize='medium')

        else:
            # Agregar la barra de color
            fig.colorbar(img, ax=ax, label='', pad=0.05, shrink=0.5, extend='both', ticks=[-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3])

        # Configurar título
        mes_term = "mes" if int(lapso) == 1 else "meses"
        fechastr = fecha.split("_")[1] +"-"+ fecha.split("_")[0]
        plt.title(f"{dictIndices[indice]} {lapso} {mes_term}  \nPeriodo: {fechastr}")
        
        # Agregar grid
        plt.draw()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Guardar la figura
        plt.savefig(fig_name, bbox_inches='tight', pad_inches=0,format='png',dpi=120)
        plt.close()
        print(f'Imagen PNG generada en: {fig_name}')
    except IOError:
        print(f"El archivo '{raster_url}' no existe o no se pudo abrir.")
    except Exception as e:
    # This will catch any other exception not caught by the specific handlers above
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

# Cargar shapefile
#shp = gpd.read_file("D:/TEMP ILIANA/Base de datos regional y Monitor de sequía/Monitor Guajira/map_shapes/nxprovinciaswgs84.shp")
shp = gpd.read_file("/root/Mon_seq/ecuador_shape/nxprovinciaswgs84.shp")

plot_raster(raster_url=nameTif, gdf=shp, fig_name=filepng, xmin=xmin, mmax=xmax, ymin=ymin, ymax=ymax, lapso=lapso, indice=indice, fecha=fecha, suavizar=True)




















