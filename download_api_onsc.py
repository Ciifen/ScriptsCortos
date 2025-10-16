""" 
El script contiene funciones que permiten descargar datos de precipitaciÃ³n y temperatura desde la API ONSC de SENAMHI
Existe tres funciones principales:
- generar_archivo_txt_mensual y  generar_archivo_txt_diario: Genera txt con datos de precipitaciÃ³n o temperatura,
  la diferencia entre las dos funciones es la temporalidad,  puede ser mensual o diario. Hay que seleccionar fecha inicial y final.
- generate_estaciones: La funciÃ³n permite generar un dataframe y guardar el excel con este dataframe conla informaciÃ³n de las estaciones
- generar_excel_estaciones_full: La funciÃ³n permite generar dos archivos csv, uno de temperatura media y otro de precipitaciÃ³n, 
  con datos diarios a partir de un listado de los cÃ³digos de estaciones de SENAMHI.

Existen dos funciones complementarias datos_vacios y read_data.
"""


"""Librerias necesarias"""
import json
import requests
import pandas as pd
"""Fin de librerias necesarias"""


"""Descripción de los parámetros de la API
Parametros de la API
 data: mensuales , diarios
 var: código(s) de variable
 cod: código(s) de estación senamhimet
 start: fecha inicial (debe proporcionarse un rango de fechas)
 end: fecha final ( debe proporcionarse un rango de fechas )
 Ejemplos
 https://onsc.senamhi.gob.bo/senamhiback/api/met/datarow?data=mensuales&var=4&cod=02033&start=2022-01-02&end=2022-05-31
 https://onsc.senamhi.gob.bo/senamhiback/api/met/datarow?data=mensuales&var=1,2,3,4,9&cod=02033,02397,02394,07050&start=2022-01-02&end=2022-05-31
"""

def datos_vacios(data):
    """La función datos_vacios valida que la lista es vacia

    Args:
        data ([list]): [Lista de datos]

    Returns:
        [boolean]
    """
    if len(data) == 0:
        return True
    else:
        return False
def read_data(url):
    """La función read_data permite descargar los datos a partir de una url

    Args:
        url ([str]): [Corresponde a la url de dónde se descargan los datos]

    Returns:
        [str or list]: [Si existen datos, retorna la lista con los datos, pero si no hay datos o algún error, retorna el texto con el error]
    """
    try:
        response_API = requests.get(url)
        data = response_API.text
        parse_json = json.loads(data)
        if not parse_json["succes"]:
            return f"Error: {parse_json['message']}"
        return parse_json["data"]
    except Exception as e:
        return f"Error en la solicitud: {str(e)}"


def generar_archivo_txt_mensual(codigo_estacion, var, fecha_inicial, fecha_final, ruta):
    """La función generar_archivo_txt genera un archivo txt con las columnas de año, mes, valor
    Args:
        codigo_estacion ([str]): [Código de la estación de SENAMHI]
        var ([int]): [Admite un número, hay dos opciones, se admite var=1 (Precipitación) o var=5 (Temperatura Media)]
        fecha_inicial ([str]): [Cadena con formato de fecha inicial "YY-MM-DD"]
        fecha_final ([str]): [Cadena con formato de fecha final "YY-MM-DD"]
    Returns:
        [str]: [Texto anunciado si se generó o no el archivo]
    """
    # Determinar la variable en función del código var
    if var == 1:
        var_completo = 1 #Var completo se refiere al codigo con el que se van a descargar los datos
        variable = "Precipitación"
        var_name = "Pcp"
    elif var == 5:
        var_completo = '3,4,5' #En Tmed, se descarga tambien Tmax y Tmin
        variable = "Temperatura Media"
        var_name = "Tmed"
    else:
        return "Variable no reconocida."
    
    # Construir la URL con los parámetros
    url = f'https://onsc.senamhi.gob.bo/senamhiback/api/met/data?data=mensuales&var={var_completo}&cod={codigo_estacion}&start={fecha_inicial}&end={fecha_final}'
    # Obtener los datos desde la API
    data = read_data(url)
    # Verificar si hubo errores en la solicitud
    if isinstance(data, str) and data.startswith("Error"):
        return data  # Devuelve el mensaje de error
    if not datos_vacios(data):
        registros = []
        if var == 5:
            # Diccionario para agrupar Tmax y Tmin por (año, mes)
            temp_dict = {}
            
            for linea in data:
                year = linea.get('year')
                month = linea.get('month')
                
                # Crear clave única para cada año-mes
                clave = (year, month)
                
                # Inicializar el registro si no existe
                if clave not in temp_dict:
                    temp_dict[clave] = {'tmax': None, 'tmin': None, 'tmed': None}
                
                # Verificar si hay Temperatura Media directa
                tmed = linea.get(variable)
                if tmed is not None:
                    temp_dict[clave]['tmed'] = tmed
                
                # Capturar Temperatura Máxima
                tmax = linea.get('Temperatura Máxima')
                if tmax is not None:
                    temp_dict[clave]['tmax'] = tmax
                
                # Capturar Temperatura Mínima
                tmin = linea.get('Temperatura Mínima')
                if tmin is not None:
                    temp_dict[clave]['tmin'] = tmin
            
            # Procesar el diccionario para calcular valores finales
            for (year, month), temps in temp_dict.items():
                if temps['tmed'] is not None:
                    valor = temps['tmed']
                # Calcular si tenemos ambas temperaturas
                elif temps['tmax'] is not None and temps['tmin'] is not None:
                    valor = (temps['tmax'] + temps['tmin']) / 2
                    valor = round(valor,3)
                else:
                    valor = None
                
                registros.append([year, month, valor])
        
        else:  # Para precipitación (var == 1)
            for linea in data:
                year = linea.get('year')
                month = linea.get('month')
                valor = linea.get(variable)
                registros.append([year, month, valor])
        
        # Convertir la lista de registros en un DataFrame
        df = pd.DataFrame(registros, columns=['Año', 'Mes', variable])
        
        # Ordenar por fechas
        df = df.sort_values(by=['Año', 'Mes'])
        
        # Definir el nombre del archivo
        nombre_archivo = f"{codigo_estacion}-{var_name}.txt"
        
        # Guardar el DataFrame en un archivo .txt
        df.to_csv(ruta+nombre_archivo, sep='\t', index=False, header=False)
        return f"Archivo {nombre_archivo} generado correctamente."


def generar_archivo_txt_diario(codigo_estacion, var, fecha_inicial, fecha_final, ruta):
    """La función generar_archivo_txt genera un archivo txt con las columnas de año, mes, dia, valor
    Args:
        codigo_estacion ([str]): [Código de la estación de SENAMHI]
        var ([int]): [Admite un número, hay dos opciones, se admite var=1 (Precipitación) o var=5 (Temperatura Media)]
        fecha_inicial ([str]): [Cadena con formato de fecha inicial "YY-MM-DD"]
        fecha_final ([str]): [Cadena con formato de fecha final "YY-MM-DD"]

    Returns:
        [str]: [Texto anunciado si se generó o no el archivo]
    """
    # Determinar la variable en función del código var
    if var == 1:
        var_completo = 1 #Var completo se refiere al codigo con el que se van a descargar los datos
        variable = "Precipitación"
        var_name = "Pcp"
    elif var == 5:
        var_completo = '3,4,5' #En Tmed, se descarga tambien Tmax y Tmin
        variable = "Temperatura Media"
        var_name = "Tmed"
    else:
        return "Variable no reconocida."
    
    # Construir la URL con los parámetros
    url = f'https://onsc.senamhi.gob.bo/senamhiback/api/met/data?data=diarios&var={var_completo}&cod={codigo_estacion}&start={fecha_inicial}&end={fecha_final}'
    print(url)
    # Obtener los datos desde la API
    data = read_data(url)
    print(data)
    # Verificar si hubo errores en la solicitud
    if isinstance(data, str) and data.startswith("Error"):
        return data  # Devuelve el mensaje de error
    
    
    if not datos_vacios(data): 
        # Crear un DataFrame a partir de los datos obtenidos
        registros = []

        if var == 5: 
            # Diccionario para agrupar Tmax y Tmin por (año, mes)
            temp_dict = {}

            for linea in data:
                year = linea.get('year')
                month = linea.get('month')
                day = linea.get('day')

                # Crear clave única para cada año-mes
                clave = (year, month)

                # Inicializar el registro si no existe
                if clave not in temp_dict:
                    temp_dict[clave] = {'tmax': None, 'tmin': None, 'tmed': None}
                
                # Verificar si hay Temperatura Media directa
                tmed = linea.get(variable)
                if tmed is not None:
                    temp_dict[clave]['tmed'] = tmed
                
                # Capturar Temperatura Máxima
                tmax = linea.get('Temperatura Máxima')
                if tmax is not None:
                    temp_dict[clave]['tmax'] = tmax
                
                # Capturar Temperatura Mínima
                tmin = linea.get('Temperatura Mínima')
                if tmin is not None:
                    temp_dict[clave]['tmin'] = tmin

                
                # Procesar el diccionario para calcular valores finales
            for (year, month, day), temps in temp_dict.items():
                # Priorizar Temperatura Media si existe
                if temps['tmed'] is not None:
                    valor = temps['tmed']
                # Calcular si tenemos ambas temperaturas
                elif temps['tmax'] is not None and temps['tmin'] is not None:
                    valor = (temps['tmax'] + temps['tmin']) / 2
                    valor = round(valor, 3)
                else:
                    valor = None

                registros.append([year, month, day, valor])
 
        else: # Para precipitación (var == 1)
            for linea in data:
                year = linea.get('year')
                month = linea.get('month')
                day = linea.get('day')
                valor = linea.get(variable)
                registros.append([year, month, day ,valor])

        # Convertir la lista de registros en un DataFrame
        df = pd.DataFrame(registros, columns=['Año', 'Mes', "Dia",variable])

        #Ordear por fechas
        df = df.sort_values(by=['Año', 'Mes', "Dia"])
            
        # Definir el nombre del archivo
        nombre_archivo = f"{codigo_estacion}-{var_name}.txt"
            
        # Guardar el DataFrame en un archivo .txt
        df.to_csv(ruta+nombre_archivo, sep='\t', index=False, header=False)
        return f"Archivo {nombre_archivo} generado correctamente."

""" Ejemplo de uso de la funcion  """
codigo_estacion = '02033'
var = 5  # Se admite var=1 (Precipitación) o var=5 (Temperatura Media)
fecha_inicial = '1990-01-01'
fecha_final = '2009-03-31'
ruta = 'D:/CIIFEN/monitor sequias osa/script download/'
#resultado = generar_archivo_txt_mensual(codigo_estacion, var, fecha_inicial, fecha_final, ruta) 
""" Fin del ejemplo"""

 
def generate_estaciones(estaciones, ruta_guardado=''):
    """La funcion generate_estaciones genera un archivo excel a partir de la api de estaciones y retorna el dataframe
    Args:
        estaciones ([str]): [URL API estaciones]
    Returns:
        [dataframe]: [Retorna un dataframe con la descrición de las estaciones]
    """
    DataEstaciones = read_data(estaciones)
    registros = []
    for linea in DataEstaciones:
        codeSenamhimet = linea.get('codeSenamhimet')
        station = linea.get('station')
        latitude = linea.get('latitude')
        longitude = linea.get('longitude')
        altitude = linea.get('altitude')
        active = linea.get('active')
        status = linea.get('status')
        registros.append([codeSenamhimet, station,latitude , longitude, altitude, active, status])
    df = pd.DataFrame(registros, columns=['codeSenamhimet', 'station', 'latitude', 'longitude', 'altitude', 'active', 'status'])
    if ruta_guardado:
        df.to_excel(ruta_guardado, index=False)
    return df

"""Ejemplo de uso"""
# estaciones ="https://onsc.senamhi.gob.bo/senamhiback/api/met/stations"
# ruta = f"D://.xlsx"
# df_estaciones = generate_estaciones(estaciones, ruta)
"""Fin de ejemplo"""


def generar_excel_estaciones_full(codigos, fecha_inicial, fecha_final):
    """La función generar_excel_estaciones_full permite generar dos archivos csv para precipitación y temperatura media,
       apartir de un listado de códigos de estaciones.

    Args:
        codigos ([list]): [Una lista de codigos en formato string]
        fecha_inicial ([str]): [Cadena con formato de fecha "YY-MM-DD"]
        fecha_final ([str]): [Cadena con formato de fecha "YY-MM-DD"]
    """

    # Dataframe vacio pcp y tmed
    dfull_med = pd.DataFrame({"year":[],"month":[],'day':[]})
    dfull_pcp = pd.DataFrame({"year":[],"month":[],'day':[]})

    # Recorrer el listado de codigos
    for codigo_estacion in codigos:
        print(f"Procesando estación: {codigo_estacion}")
        
        # Construir la URL con los parámetros
        url = f'https://onsc.senamhi.gob.bo/senamhiback/api/met/data?data=diarios&var=1,3,4,5&cod={codigo_estacion}&start={fecha_inicial}&end={fecha_final}'
        
        # Obtener los datos desde la API
        data = read_data(url)
        
        # Verificar si hubo errores en la solicitud
        if isinstance(data, str) and data.startswith("Error"):
            print(f"Error en {codigo_estacion}: {data}")
            continue  # Continuar con la siguiente estación
        
        if not datos_vacios(data):
            # Diccionarios para agrupar por fecha
            temp_dict = {}
            pcp_dict = {}
            
            for linea in data:
                year = linea.get('year')
                month = linea.get('month')
                day = linea.get('day')
                
                # Crear clave única para cada fecha
                clave = (year, month, day)
                
                # Procesar Precipitación
                pcp = linea.get("Precipitación")
                if pcp is not None:
                    pcp_dict[clave] = pcp
                
                # Procesar Temperatura
                if clave not in temp_dict:
                    temp_dict[clave] = {'tmax': None, 'tmin': None, 'tmed': None}
                
                tmed = linea.get("Temperatura Media")
                if tmed is not None:
                    temp_dict[clave]['tmed'] = tmed
                
                tmax = linea.get('Temperatura Máxima')
                if tmax is not None:
                    temp_dict[clave]['tmax'] = tmax
                
                tmin = linea.get('Temperatura Mínima')
                if tmin is not None:
                    temp_dict[clave]['tmin'] = tmin
            
            # Crear listas de registros procesados
            registros_pcp = []
            registros_tmed = []
            
            # Obtener todas las fechas únicas
            todas_fechas = set(list(temp_dict.keys()) + list(pcp_dict.keys()))
            
            for (year, month, day) in sorted(todas_fechas):
                # Procesar Precipitación
                pcp_valor = pcp_dict.get((year, month, day))
                registros_pcp.append([year, month, day, pcp_valor])
                
                # Procesar Temperatura
                temps = temp_dict.get((year, month, day), {})
                if temps.get('tmed') is not None:
                    tmed_valor = temps['tmed']
                elif temps.get('tmax') is not None and temps.get('tmin') is not None:
                    tmed_valor = (temps['tmax'] + temps['tmin']) / 2
                else:
                    tmed_valor = None
                
                registros_tmed.append([year, month, day, tmed_valor])
            
            # Convertir a DataFrames solo después de agrupar
            df_tmed = pd.DataFrame(registros_tmed, columns=['year', 'month', "day", codigo_estacion])
            df_pcp = pd.DataFrame(registros_pcp, columns=['year', 'month', "day", codigo_estacion])
            
            # Unir con el dataframe general
            dfull_med = pd.merge(dfull_med, df_tmed, how="outer", on=['year','month','day'])
            dfull_pcp = pd.merge(dfull_pcp, df_pcp, how="outer", on=['year','month','day'])
            
            print(f"Estación {codigo_estacion} procesada. Registros Tmed: {len(df_tmed)}, Pcp: {len(df_pcp)}")
    
    # Ordenar los dataframes finales
    dfull_med = dfull_med.sort_values(by=['year', 'month', 'day']).reset_index(drop=True)
    dfull_pcp = dfull_pcp.sort_values(by=['year', 'month', 'day']).reset_index(drop=True)
    
    # Guardar el DataFrame en un archivo general
    nombre_archivo_tmed = "D://CIIFEN//monitor sequias osa//script download//estaciones_tmed_full.csv"
    nombre_archivo_pcp = "D://CIIFEN//monitor sequias osa//script download//estaciones_pcp_full.csv"
    
    dfull_med.to_csv(nombre_archivo_tmed, index=False, header=True)
    dfull_pcp.to_csv(nombre_archivo_pcp, index=False, header=True)
    
    print(f"\nArchivos generados exitosamente:")
    print(f"Tmed: {nombre_archivo_tmed} - Shape: {dfull_med.shape}")
    print(f"Pcp: {nombre_archivo_pcp} - Shape: {dfull_pcp.shape}")

"""Ejemplo de uso donde se descargan los datos desde 1990 y se filtra las estaciones en estado activo"""
fecha_inicial = "1990-01-01"
fecha_final = "2024-10-10"
estaciones ="https://onsc.senamhi.gob.bo/senamhiback/api/met/stations"
df_estaciones = generate_estaciones(estaciones)
df_estaciones_activo = df_estaciones[df_estaciones["status"]=='Activo']
list_cod_estaciones = df_estaciones_activo['codeSenamhimet'].to_list()
#generar_excel_estaciones_full(list_cod_estaciones, fecha_inicial, fecha_final)
"""Fin del ejemplo"""




