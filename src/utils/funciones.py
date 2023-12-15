
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import kurtosis, skew, probplot, shapiro, spearmanr, kruskal
from geopy.geocoders import Nominatim
import time
import pandas as pd
import plotly.express as px

pd.set_option('display.max_columns', None)


def nombre_columnas(data):
    '''Función para tratar el nombre de las columnas y eliminar las vacias'''
    try:
        data.drop(columns=['latitude', 'longitude', 'portal', 'door', 'rent_price_by_area', 'are_pets_allowed', 'is_furnished',
                    'is_kitchen_equipped', 'has_private_parking', 'has_public_parking', 'sq_mt_useful', 'n_floors', 'has_ac', 'title',
                    'sq_mt_allotment', 'raw_address', 'is_exact_address_hidden', 'street_name', 'street_number', 'is_buy_price_known',
                    'is_parking_included_in_price', 'is_rent_price_known', 'operation', 'is_new_development', 'parking_price', 'rent_price', 'id', 'neighborhood_id',
                    'has_central_heating', 'has_individual_heating', 'has_lift', 'is_orientation_east', 'is_orientation_north', 'is_orientation_south', 'is_orientation_west'
                    ], axis=1, inplace = True)
        
        data.columns = ['annio_construccion', 'precio_venta', 'precio_venta_por_m2', 'cee',
    'piso', 'balcon', 'armarios_empotrados', 'jardin', 'zonas_verdes', 
    'estacionamiento', 'pileta',
    'trastero', 'terraza', 'tipo_inmueble',
    'accesible', 'exterior', 'bajo', 'necesita_reforma', 'bannos', 'habitaciones',
        'metros_cuadrados', 'ubicacion']

    except Exception as a:
        print(f"No pude tranformar las columnas por {a}")
    return data

def cardinalidad(data):
    '''Funcion para saber la cardinalidad de las varibales que tenemos en el data frame'''
    df_cardin = pd.DataFrame([{
                'variable' : i,
                'tipo_dato' : data[i].dtypes,
                'cantidad_de_nulos' : data[i].isna().sum(),
                'valores_unicos' : data[i].unique(),
                'cardinalidad' : data[i].nunique(),
                'porcentaje_cardinalidad' : (data[i].nunique()/data.shape[0])*100
            } for i in data])
    
    df_tipo_variable = pd.DataFrame({'tipo_variable' : ['discreta', 'continua', 'continua', 'ordinal',
            'ordinal', 'nominal', 'nominal', 'nominal', 'nominal',
            'nominal', 'nominal', 'nominal', 'nominal', 'nominal',
            'nominal', 'nominal', 'nominal', 'nominal', 'discreta',
            'discreta', 'continua', 'nominal']})
    
    df_cardinalidad = pd.concat([df_cardin,df_tipo_variable], axis = 1)

    return df_cardinalidad

def graficos_variables_cuant(data):
    ''''Funcuion para graficar las variables cuantitativas'''
    media_color = 'r'
    mediana_color = 'b'
    try:
        for columna in data.columns:
            print('--'*30)
            print(f"VARIABLE: {columna}\n")

            media = data[columna].mean()
            mediana = data[columna].median()

            plt.figure(figsize=(20,4))
            sns.boxplot(data[columna], orient='h', palette='husl')
            plt.axvline(media, color = media_color, linestyle = 'dashed', linewidth = 1)
            plt.axvline(mediana, color = mediana_color, linestyle = 'dashed', linewidth = 1)

            plt.show()

            sns.displot(data[columna], rug = True, palette='husl' , bins = 30)
            plt.axvline(media, color = media_color, linestyle = 'dashed', linewidth = 1, label = f'Media: {media:.0f}')
            plt.axvline(mediana, color = mediana_color, linestyle = 'dashed', linewidth = 1, label = f'Mediana: {mediana:.0f}')

            plt.title(f'Distribución de {columna}')
            plt.legend()

            plt.show()

            print(data[columna].describe().round())
            print('--'*30)
    except Exception as a:
        print(f"No puedo analizar la variable por este error {a}")

def graficos_variables_cualit(data):
    ''''Funcuion para graficar las variables cualitativas'''
    try:
        for columna in data.columns:
            print('--'*50)
            print(f"VARIABLE: {columna}\n")
            if len(data[columna].dropna().unique()) > 1:
                ax = sns.countplot(data= data.sort_values(by=columna), x= columna, palette='husl')
                ax.set_xticklabels(data[columna].sort_values().unique(), rotation=90)
                #se superponen los valores porque me da uns advertencia al aplicar este parametro, no sé como sacarla :)
                plt.title(f"Conteo variable {columna}")
                plt.show();
            else:
                print('No es necesario graficar porque tiene un solo valor dentro de la columna')
            print(data[columna].value_counts())
            print('--'*50)
    except Exception as a:
        print(f"No puedo analizar la variable por este error {a}")

def rellenar_columnas_F(data):
    ''' Función que rellena las columnas que tienen valor true y nan'''
    try:
        data['zonas_verdes'].replace(np.nan, False,inplace=True)
        data['balcon'].replace(np.nan, False,inplace=True)
        data['armarios_empotrados'].replace(np.nan, False,inplace=True)
        data['jardin'].replace(np.nan, False,inplace=True)
        data['pileta'].replace(np.nan, False,inplace=True)
        data['trastero'].replace(np.nan, False,inplace=True)
        data['terraza'].replace(np.nan, False,inplace=True)
        data['accesible'].replace(np.nan, False,inplace=True)
    except Exception as a:
        print(f"No pude rellenar las columnas por {a}")
    return data

def rellenar_annios_nulos_necesitan_reforma(data):
    '''Función para rellenar los annios que vienen nulos'''
    try:
        #diccionario para ver si tenemos todos las ubicaciones o no
        dicc_annios_antiguos = data[(data['necesita_reforma'] == True) & (data['annio_construccion'].notna())].groupby('ubicacion')[['annio_construccion']].mean(numeric_only = True).astype(int).reset_index().to_dict('records')
        
        #agrega las ubicaciones que no existen, asignando la media de los inmuebles que necesitan reforma
        dicc_annios_antiguos = dicc_annios_antiguos + [{'ubicacion': 'Horcajo, Madrid', 'annio_construccion': 1957}, 
                   {'ubicacion': 'Valdebebas - Valdefuentes, Madrid', 'annio_construccion': 1957},
                   {'ubicacion': 'Virgen del Cortijo - Manoteras, Madrid', 'annio_construccion': 1957}]
        
        data_annios_antiguos = pd.DataFrame(dicc_annios_antiguos)
        data_unido = pd.merge(data,data_annios_antiguos, on='ubicacion', how = 'left')

        #asigna el valor de el annio en base a la la ubicacion
        data_unido['annio_construccion'] = data_unido.apply(lambda x: x.annio_construccion_y if ((x.necesita_reforma == True) & (pd.isna(x.annio_construccion_x))) else x.annio_construccion_x, axis = 1)

        data = data_unido.drop(columns=['annio_construccion_y', 'annio_construccion_x'], axis = 1)
    
    except Exception as a:
        print(f"No pude transformar el df por {a}")

    return data

def rellenar_annios_nulos_no_necesitan_reforma(data):
    '''Función para rellenar los annios que vienen nulos'''
    try:
        #diccionario para ver si tenemos todos las ubicaciones o no
        dicc_annios_nuevo = data[(data['necesita_reforma'] == False) & (data['annio_construccion'].notna())].groupby('ubicacion')[['annio_construccion']].mean(numeric_only = True).astype(int).reset_index().to_dict('records')
        
        #agrega las ubicaciones que no existen, asignando la media de los inmuebles que no necesitan reforma
        dicc_annios_nuevo = dicc_annios_nuevo + [{'ubicacion': 'Cuatro Vientos, Madrid', 'annio_construccion': 1973}]
        
        data_annios_nuevo = pd.DataFrame(dicc_annios_nuevo)
        data_unido_nuevo = pd.merge(data,data_annios_nuevo, on='ubicacion', how='left')

        #asigna el valor de el annio en base a la la ubicacion
        data_unido_nuevo['annio_construccion'] = data_unido_nuevo.apply(lambda x: x.annio_construccion_y if ((x.necesita_reforma == False) & (pd.isna(x.annio_construccion_x))) else x.annio_construccion_x, axis = 1)

        data = data_unido_nuevo.drop(columns=['annio_construccion_x', 'annio_construccion_y'], axis = 1)
    
    except Exception as a:
        print(f"No pude transformar el df por {a}")

    return data

def rellenar_annio_outlier(data):
    '''Funcion para rellenar un año de construccion incorrecto'''
    media_año_barrio_s = data[(data['ubicacion'] == 'Barrio de Salamanca, Madrid') & (data['annio_construccion'].notna())].groupby('ubicacion')['annio_construccion'].mean(numeric_only = True).astype(int)
    data['annio_construccion'].replace(8170.0, 1979, inplace= True)
    return data

def rellenar_pisos_nulos(data):
    '''Funcion para rellenar los valores nulos de los pisos, con la moda segun la ubicacion'''
    try:
        #df el piso que más se repite, respetando las alturas por ubicacion segun normativa
        df_piso_más_comun = data[data['piso'].notna()].groupby(['ubicacion', 'piso'], as_index=False).count()[['ubicacion', 'piso']].groupby('ubicacion', as_index=False).max()

        df_unido_pisos = pd.merge(data,df_piso_más_comun, on='ubicacion', how= 'inner')

        df_unido_pisos['piso'] = df_unido_pisos.apply(lambda x: x.piso_y if pd.isna(x.piso_x) else x.piso_x, axis = 1)

        data = df_unido_pisos.drop(columns=['piso_x', 'piso_y'], axis = 1)
    except Exception as a:
        print(f"No pude transformar el df por {a}")
    return data

def rellenar_bajos_nulos(data):
    '''Funcion que rellena los valores nulos en la columna bajo en base al piso en el que se encuentra'''
    try:
        bajos = ('Semi-sótano', 'Entreplanta interior', 'Entreplanta', 'Semi-sótano exterior', 'Semi-sótano interior', 'Sótano interior', 'Sótano', 'Sótano exterior')

        data['bajo'] = data['piso'].apply(lambda x: True if x in bajos else False)

    except Exception as a:
        print(f"No puse tranformar el df por {a}")
    
    return data

def sacar_metros_cuadrados_nuevos(data):
    ''''Funcion para rellenar los valores nulos de los metros cuadrados en base a el precio por metro cuadrado'''
    try:
        data.drop(columns=['metros_cuadrados'], axis= 1, inplace=True)
        data['metros_cuadrados'] = (data['precio_venta'] / data['precio_venta_por_m2']).round()
    except Exception as a:
        print(f"No pude tranformar el dataframe en la función 'sacar_metros_cuadrados_nuevos'")
    return data

def rellenar_exterior(data):
    '''Funcion que rellena los valores nulos en la columna exterior en base a el piso en el que se encuentra'''
    try:
        exteriores = ('Entreplanta exterior', 'Semi-sótano exterior', 'Sótano exterior')

        data['exterior'] = data['piso'].apply(lambda x: True if x in exteriores else False)

    except Exception as a:
        print(f"No puse tranformar el df por {a}")
    
    return data

def rellenar_tipo_inmueble(data):
    '''Funcion que rellena los valores nulos en la columna tipo_inmueble, los unico no completos son los estudios'''
    try:
        data['tipo_inmueble'].fillna(value ='HouseType 1: Pisos', inplace=True)

    except Exception as a:
        print(f"No puse tranformar el df por {a}")
    
    return data

def rellenar_bannos_nulos(data):
    '''Funcion para rellenar los valores nulos de los bannos, con la media segun los metros cuadrados'''
    try:
        dicc_bannos = data[data['bannos'].notna()].groupby(['metros_cuadrados'], as_index=False)[['bannos']].mean().round().to_dict('records')
        dicc_bannos.append({'metros_cuadrados': 661, 'bannos': 5})

        df_banno_mas_comun = pd.DataFrame(dicc_bannos)

        df_unido_bannos = pd.merge(data,df_banno_mas_comun, on='metros_cuadrados', how= 'inner')

        df_unido_bannos['bannos'] = df_unido_bannos.apply(lambda x: x.bannos_y if pd.isna(x.bannos_x) else x.bannos_x, axis = 1)

        data = df_unido_bannos.drop(columns=['bannos_x', 'bannos_y'], axis = 1)
    except Exception as a:
        print(f"No pude transformar el df por {a}")
    return data

def determinacion_zonas(data):
    '''Funcion para subdivir las localizaciones por zonas'''
    try:
        sur = ['Usera, Madrid', 'Puente de Vallecas, Madrid', 'Carabanchel, Madrid', 'Villaverde, Madrid', 'Puerta Bonita, Madrid', 'Vista Alegre, Madrid', 'San Fermín, Madrid', 'Pradolongo, Madrid', '12 de Octubre-Orcasur, Madrid', 'Almendrales, Madrid', 'Moscardó, Madrid', 'Zofío, Madrid', 'Los Ángeles, Madrid', 'San Cristóbal, Madrid']
        este = ['Vicálvaro, Madrid', 'Casco Histórico de Vallecas, Madrid', 'Ensanche de Vallecas - La Gavia, Madrid', 'Santa Eugenia, Madrid', 'Orcasitas, Madrid', 'San Diego, Madrid', 'Valdebernardo - Valderribas, Madrid', 'Valdezarza, Madrid', 'Barajas, Madrid', 'Arapiles, Madrid', 'San Juan Bautista, Madrid', 'Prosperidad, Madrid', 'Ciudad Lineal, Madrid', 'Costillares, Madrid', 'Pueblo Nuevo, Madrid', 'Quintana, Madrid']
        centro = ['Retiro, Madrid', 'Arganzuela, Madrid', 'Chamberí, Madrid', 'Centro, Madrid', 'Malasaña-Universidad, Madrid', 'Palacio, Madrid', 'Sol, Madrid', 'Chueca-Justicia, Madrid', 'Huertas-Cortes, Madrid', 'La Paz, Madrid', 'Recoletos, Madrid', 'Jerónimos, Madrid', 'Atalaya, Madrid', 'Niño Jesús, Madrid', 'Fuentelarreina, Madrid', 'Alameda de Osuna, Madrid', 'Media Legua, Madrid']
        oeste = ['Moncloa, Madrid', 'Chamartín, Madrid', 'Tetuán, Madrid', 'Argüelles, Madrid', 'Valdemarín, Madrid', 'Ciudad Universitaria, Madrid', 'Nuevos Ministerios-Ríos Rosas, Madrid', 'Aravaca, Madrid', 'Vallehermoso, Madrid', 'Cuatro Caminos, Madrid', 'Ventilla-Almenara, Madrid', 'Sanchinarro, Madrid', 'El Viso, Madrid', 'Ciudad Jardín, Madrid', 'Chopera, Madrid', 'Valdemarín, Madrid', 'Virgen del Cortijo - Manoteras, Madrid']
        norte = ['Fuencarral, Madrid', 'Peñagrande, Madrid', 'Pilar, Madrid', 'Pinar del Rey, Madrid', 'Canillas, Madrid', 'Tres Olivos - Valverde, Madrid', 'Conde Orgaz-Piovera, Madrid', 'Hortaleza, Madrid', 'Apóstol Santiago, Madrid', 'Nuevos Ministerios-Ríos Rosas, Madrid', 'Arapiles, Madrid', 'Bernabéu-Hispanoamérica, Madrid', 'Prosperidad, Madrid', 'Castilla, Madrid', 'Fuente del Berro, Madrid', 'Media Legua, Madrid']

        funcion_lambda = lambda x: 'sur' if x in sur else ('este' if x in este else ('centro' if x in centro else ('oeste' if x in oeste else 'norte')))
        data['zona'] = data['ubicacion'].apply(funcion_lambda)
    except Exception as a:
        print(f"No pude transformar el df por {a}")
    return data

def obtener_coordenadas(data):
    '''Funcion para asignar coordenadas a las ubicaciones que tenemos en el dataframe'''
    try: 
        
        def obtener_coordenadas(direccion):
            try:
                geolocalizador = Nominatim(user_agent="maria")
                ubicacion = geolocalizador.geocode(direccion)

                if ubicacion:
                    latitud = ubicacion.latitude
                    longitud = ubicacion.longitude

                    return latitud, longitud
                else:
                    print(f"No se encontraron coordenadas para la dirección proporcionada {direccion}.")
                    time.sleep(1)
                    latitud = float(input('Lat: '))
                    longitud = float(input('Long: '))
                    return latitud, longitud
            except:
                pass

        dicc = []
        for i in data['ubicacion'].unique():
            coodenadas = obtener_coordenadas(i)
            dicc.append({'ubicacion' : i, 
                        'latitud' : coodenadas[0],
                        'longitud' : coodenadas[1]})
        df_coordenadas = pd.DataFrame(dicc)
    except Exception as a:
        print(f"No pude generar el df por {a}")
    return df_coordenadas


def grafico_variable_ppal(data):
    ''' 

    Funcion para graficar la variable principal a analizar
    input : df
    output: grafico

    '''
    try:
        media_color = 'r'
        mediana_color = 'b'
        media = data['precio_venta_por_m2'].mean()
        median = data['precio_venta_por_m2'].median()
        variance = data['precio_venta_por_m2'].var()
        desv_std = data['precio_venta_por_m2'].std()  
        kurtosis_valor = kurtosis(data['precio_venta_por_m2'])
        simetria_valor = skew(data['precio_venta_por_m2'])

        sns.kdeplot(data=data, x='precio_venta_por_m2',fill=True,palette='hls')
        # Agregar líneas verticales para las estadísticas
        plt.axvline(media, color=media_color, linestyle='dashed', label=f'Media: {media:.2f}')
        plt.axvline(median, color= mediana_color, linestyle='dashed', label=f'Median: {median:.2f}')
        plt.axvline(desv_std, color='y', linestyle='dashed', label=f'Desv_std: {desv_std:.2f}')

        plt.title('Distribución del precio por metros cuadrados')
        # plt.xlabel('Popularidad')
        # plt.ylabel('Densidad')

        plt.legend()

        plt.show()
        # Interpretación de los valores

        print(f"kurtosis: {kurtosis_valor:.2f}")
        print(f"simetria: {simetria_valor:.2f}")

        if kurtosis_valor > 0:
            print("La distribución es leptocúrtica, lo que sugiere colas pesadas y picos agudos.")
        elif kurtosis_valor < 0:
            print("La distribución es platicúrtica, lo que sugiere colas ligeras y un pico achatado.")
        else:
            print("La distribución es mesocúrtica, similar a una distribución normal.")

        if simetria_valor < 0:
            print("La distribución es asimétrica positiva (sesgo hacia la derecha).")
        elif simetria_valor > 0:
            print("La distribución es asimétrica negativa (sesgo hacia la izquierda).")
        else:
            print("La distribución es perfectamente simétrica alrededor de su media.")
    except Exception as a:
        print(f"No pude analizar la variable por {a}")

def prueba_grafica_normalidad(data):
    '''Función para analizar graficamente si una variable tiene una distribución normal'''
    try: 
        probplot(data['precio_venta_por_m2'], dist="norm", plot=plt)
        plt.title(f'Q-Q Plot de Precio de venta por m2')
        plt.show()
    except Exception as a:
        print(f"No pude analizar la variable por {a}")

def prueba_normalidad_shapiro(data):
    '''Función para analizar si una variable tiene una distribución normal mediante shapiro'''
    try:
        stat, p_value = shapiro(data['precio_venta_por_m2'])
        print("Estadística de prueba:", stat)
        print("Valor p:", p_value)

        if p_value < 0.05:
            print("Rechazamos la hipótesis nula; los datos no siguen una distribución normal.")
        else:
            print("No hay suficiente evidencia para rechazar la hipótesis nula; los datos podrían seguir una distribución normal.")
    except Exception as a:
        print(f"No pude analizar la variable por {a}")

def pair_plot(data):
    '''Funcion para ver graficamente como se comportan algunas de  las variables cuantitativas'''
    try:
        df_cuant_pair_plot = data.select_dtypes(include = 'number').drop(columns=['annio_construccion', 'latitud', 'longitud'], axis=1)
        sns.pairplot(df_cuant_pair_plot, kind='reg', palette='husl', markers='.');
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def precio_cee(data):
    '''Función para ver graficamente la relación que existe entre la letra del certificado energético y el precio de venta por metro cuadrado'''
    try: 
        df_precio_venta_cee = data.groupby('cee', as_index=False).mean(numeric_only = True)
        ax = sns.catplot(x = 'precio_venta_por_m2', y='cee', hue = 'cee', kind= 'bar',
        data=df_precio_venta_cee.sort_values(by='cee'), palette='husl');
        # ax.set_xticklabels(df_precio_venta_cee['cee'].sort_values().unique(), rotation=90)
        plt.title('Relación entre CEE y Precio de venta por metros cuadrados')
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def precio_tipo_inmueble(data):
    '''Función para ver graficamente la relación que existe entre el tipo de inmueble y el precio de venta por metro cuadrado'''
    try: 
        df_precio_venta_tipo_inmueble = data.groupby('tipo_inmueble', as_index=False, sort=True).mean(numeric_only = True)
        ax = sns.catplot(x= 'precio_venta_por_m2', y = 'tipo_inmueble', data = df_precio_venta_tipo_inmueble, kind='bar', hue = 'tipo_inmueble', palette='husl')
        # ax.set_xticklabels(df_precio_venta_tipo_inmueble['tipo_inmueble'].sort_values().unique(), rotation = -45)
        plt.title('Relación entre tipo de inmueble y Precio de venta por metros cuadrados')
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def tranformacion_numerica(data):
    '''Función para convertir los booleanos que tenemos en el dataframe en 1 y 0 para poder analizar otras cosas'''
    try:
        df_todo_n = data.copy()
        df_todo_n.replace(False, 0, inplace=True)
        df_todo_n.replace(True, 1, inplace=True)
    except Exception as a:
        print(f"No pude analizar la variable por {a}")
    return df_todo_n

def grafico_precio_var1_var2(data, var1, var2):
    '''Función para evaluar como aumenta el precio de venta por m2 respecto de las habitaciones y otra variable cualitativa a elección
    Input: 
    data = dataframe
    variable = columa dataframe
    '''
    try:
        sns.scatterplot(x= var1, y = 'precio_venta_por_m2', data = data, hue = var2, palette='husl')
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def grafico_precio_zona_yvariable(data, variable):
    '''Función para evaluar como aumenta el precio de venta por m2 por zona y otra variable a elegir
    Input: 
    data = dataframe
    variable = columa dataframe
    '''
    try:
        ax = sns.catplot(x = 'precio_venta_por_m2', y='zona', hue = variable, kind= 'bar', palette='husl',
            data=data, errorbar = 'sd', err_kws={'linewidth': 1});
        # ax.set_xticklabels(data['zona'].sort_values().unique(), rotation = -45)
        plt.title(f'Relación entre {variable} y precio de venta por m2 por Zonas')
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def grafico_precio_zona(data):
    '''Función para evaluar como aumenta el precio de venta por m2 por zona y otra variable a elegir
    Input: 
    data = dataframe
    variable = columa dataframe
    '''
    try:
        ax = sns.catplot(x = 'precio_venta_por_m2', y='zona', hue = 'zona', kind= 'bar', palette='husl',
            data=data, errorbar = 'sd', err_kws={'linewidth': 1});
        # ax.set_xticklabels(data['zona'].sort_values().unique(), rotation = -45)
        plt.title(f'Relación entre la zona y precio de venta por m2')
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def grafico_precio_var1(data, variable):
    '''Función para ver graficamente la relación que existe entre la letra del certificado energético y el precio de venta por metro cuadrado'''
    try: 
        df_precio_venta_var = data.groupby(variable, as_index=False).mean(numeric_only = True)
        ax = sns.catplot(y = 'precio_venta_por_m2', x=variable, hue = variable, kind= 'bar',
        data=df_precio_venta_var, palette= 'husl')
        # ax.set_xticklabels(df_precio_venta_cee[variable].sort_values().unique(), rotation=90)
        plt.title(f'Relación entre {variable} y Precio de venta por metros cuadrados')
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def grafico_mapa(data):
    '''Función para graficar en un mapa las variables de precio venta por m2, por zona y según el tamaño que tienen'''
    try:
        fig = px.scatter_mapbox(data, lat = 'latitud', lon = 'longitud', size = 'metros_cuadrados',
                        color = 'precio_venta_por_m2', color_continuous_scale = 'plasma',
                        zoom = 3, mapbox_style = 'open-street-map')  
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def grafico_heatmap(data):
    '''Función para graficar en un mapa de calor mostrando las correlaciones entre las variables'''
    try:
        df_cuant = data.select_dtypes(include = 'number')
        plt.figure(figsize=(10,10))
        sns.heatmap(df_cuant.corr(numeric_only=True), robust=True, 
                    square = True, linewidths = .3)      
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def grafico_var1_var2(data, var1, var2):
    '''Función para graficar la relacion de dos variables'''
    try:
        plt.figure(figsize=(10, 6))
        sns.regplot(x=var1, y=var2, data=data, marker='o')

        plt.title(f'Relación entre {var1} y {var2}')
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def prueba_corr_pearson(df, var1, var2):
    try:
        correlation_coefficient, p_value = spearmanr(df[var1], df[var2])
        print(f"Coeficiente de correlación de Spearman: {correlation_coefficient}")
        print(f"Valor p: {p_value}")

        alpha = 0.05
        if p_value < alpha:
            print("Hay evidencia para rechazar la hipótesis nula; existe una correlación significativa.")
        else:
            print("No hay suficiente evidencia para rechazar la hipótesis nula; no se puede afirmar una correlación significativa.")
    except Exception as a:
        print(f"No pude evaluar la correlación por {a}")

def prueba_krus_cee(data):
    # Prueba de Kruskal-Wallis para más de dos muestras independientes
    stat_kw, p_value_kw = kruskal(data['precio_compra_por_m2'][data['cee'] == 'A'],
                                data['precio_compra_por_m2'][data['cee'] == 'B'],
                                data['precio_compra_por_m2'][data['cee'] == 'C'],
                                data['precio_compra_por_m2'][data['cee'] == 'D'],
                                data['precio_compra_por_m2'][data['cee'] == 'E'],
                                data['precio_compra_por_m2'][data['cee'] == 'F'],
                                data['precio_compra_por_m2'][data['cee'] == 'G'],
                                data['precio_compra_por_m2'][data['cee'] == 'inmueble exento'],
                                data['precio_compra_por_m2'][data['cee'] == 'no indicado'],
                                data['precio_compra_por_m2'][data['cee'] == 'en trámite']
                                )
    alpha = 0.05 
    # Hipótesis nula (H0): No hay diferencia significativa en la calificación entre las letras de los certificados.
    # Hipótesis alternativa (Ha): Existe al menos una diferencia significativa en la calificación entre las letras de los certificados.

    print(f"\nPrueba de Kruskal-Wallis para más de dos muestras independientes: stat = {stat_kw}, p_value = {p_value_kw}")

    if p_value_kw < alpha:
        print("Rechazamos la hipótesis nula. Hay evidencia de al menos una diferencia significativa en la calificación entre las letras de los certificados")
    else:
        print("No hay suficiente evidencia para rechazar la hipótesis nula. No hay diferencia significativa en la calificación entre las letras de los certificados.")