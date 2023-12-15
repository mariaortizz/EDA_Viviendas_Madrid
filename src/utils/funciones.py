class EDA_funciones:
    def __init__(self):
        pass 

    def nombre_columnas(self,df):
        '''Función para tratar el nombre de las columnas y eliminar las vacias'''
        try:
            df.drop(columns=['latitude', 'longitude', 'portal', 'door', 'rent_price_by_area', 'are_pets_allowed', 'is_furnished',
                        'is_kitchen_equipped', 'has_private_parking', 'has_public_parking', 'sq_mt_useful', 'n_floors', 'has_ac', 'title',
                        'sq_mt_allotment', 'raw_address', 'is_exact_address_hidden', 'street_name', 'street_number', 'is_buy_price_known',
                        'is_parking_included_in_price', 'is_rent_price_known', 'operation', 'is_new_development', 'parking_price', 'rent_price', 'id', 'neighborhood_id',
                        'has_central_heating', 'has_individual_heating', 'has_lift', 'is_orientation_east', 'is_orientation_north', 'is_orientation_south', 'is_orientation_west'
                        ], axis=1, inplace = True)
            
            df.columns = ['annio_construccion', 'precio_compra', 'precio_compra_por_m2', 'cee',
        'piso', 'balcon', 'armarios_empotrados', 'jardin', 'zonas_verdes', 
        'estacionamiento', 'pileta',
        'trastero', 'terraza', 'tipo_inmueble',
        'accesible', 'exterior', 'bajo', 'necesita_reforma', 'bannos', 'habitaciones',
            'metros_cuadrados', 'ubicacion']

        except Exception as a:
            print(f"No pude tranformar las columnas por {a}")
        return df

    def cardinalidad(self,df):
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