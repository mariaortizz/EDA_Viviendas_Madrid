
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

pd.set_option('display.max_columns', None)


def nombre_columnas(df):
    '''Función para tratar el nombre de las columnas y eliminar las vacias'''
    try:
        df.drop(columns=['latitude', 'longitude', 'portal', 'door', 'rent_price_by_area', 'are_pets_allowed', 'is_furnished',
                    'is_kitchen_equipped', 'has_private_parking', 'has_public_parking', 'sq_mt_useful', 'n_floors', 'has_ac', 'title',
                    'sq_mt_allotment', 'raw_address', 'is_exact_address_hidden', 'street_name', 'street_number', 'is_buy_price_known',
                    'is_parking_included_in_price', 'is_rent_price_known', 'operation', 'is_new_development', 'parking_price', 'rent_price', 'id', 'neighborhood_id',
                    'has_central_heating', 'has_individual_heating', 'has_lift', 'is_orientation_east', 'is_orientation_north', 'is_orientation_south', 'is_orientation_west'
                    ], axis=1, inplace = True)
        
        df.columns = ['annio_construccion', 'precio_venta', 'precio_venta_por_m2', 'cee',
    'piso', 'balcon', 'armarios_empotrados', 'jardin', 'zonas_verdes', 
    'estacionamiento', 'pileta',
    'trastero', 'terraza', 'tipo_inmueble',
    'accesible', 'exterior', 'bajo', 'necesita_reforma', 'bannos', 'habitaciones',
        'metros_cuadrados', 'ubicacion']

    except Exception as a:
        print(f"No pude tranformar las columnas por {a}")
    return df

def cardinalidad(df):
    '''Funcion para saber la cardinalidad de las varibales que tenemos en el data frame'''
    df_cardin = pd.DataFrame([{
                'variable' : i,
                'tipo_dato' : df[i].dtypes,
                'cantidad_de_nulos' : df[i].isna().sum(),
                'valores_unicos' : df[i].unique(),
                'cardinalidad' : df[i].nunique(),
                'porcentaje_cardinalidad' : (df[i].nunique()/df.shape[0])*100
            } for i in df])
    
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
        data['metros_cuadrados'] = (data['precio_compra'] / data['precio_compra_por_m2']).round()
    except Exception as a:
        print(f"No pude tranformar el dataframe")
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


