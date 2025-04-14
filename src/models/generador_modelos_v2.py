import math
import yaml
from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Concatenate, Flatten, RepeatVector # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from procesador_datos import DataManager # type: ignore


class HybridModelBuilder():
    """
    Hybrid_model es una clase que construye un modelo híbrido utilizando embeddings categóricos, 
    LSTM para secuencias temporales y entradas numéricas para variables de predicción.

    Args:
        categorical_features (list of dict): Lista de variables categóricas.
            Cada diccionario debe contener:
            - 'name': Nombre de la variable categórica (str).
            - 'num_categories': Número total de categorías únicas en la variable (int).
        timesteps (int): Cantidad de pasos temporales en las secuencias.
        var_numericas (int): Número de variables numéricas por muestra.
    """
    def __init__(self, metadata):
        """
        Inicializa los parámetros del modelo, incluyendo las variables categóricas, 
        los pasos temporales y las variables numéricas.

        Args:
            categorical_features (list of dict): Variables categóricas con nombre y número de categorías.
            timesteps (int): Número de pasos temporales.
            var_numericas (int): Número de variables numéricas por muestra.
        """
        self.categorical_features = metadata['categorical']
        self.timesteps = self._load_config()['timesteps']
        self.var_numericas = metadata['var_num']
        self.cat_inputs = []
        self.cat_embeddings = []
    
    def _load_config(self):
        with open('project_config.yaml', "r") as f:
            return yaml.safe_load(f)
    
    def calcular_output_dim(self, input_dim, d=2):
        output_dim = math.ceil(d * math.log(input_dim))  # Redondea hacia arriba
        return output_dim

    
    def crear_embedding_module(self, name, n_categorias):
        output_dim = self.calcular_output_dim(n_categorias)
        model = Sequential([
            Embedding(input_dim=n_categorias+1, output_dim=output_dim, name=name+'_embed'),
            Flatten(),
            RepeatVector(self.timesteps),
            ],
            name = 'embedding_module_' + name)
        return model
    
    def create_categorical_modules(self):
        for key in self.categorical_features.keys():
            # Crea el input para la variable categorica
            input_layer = Input(shape=(1,), name=key + '_input') 
            self.cat_inputs.append(input_layer)

            # Crea el embedding asociado
            embedding_layer = self.crear_embedding_module(key, self.categorical_features[key])
            self.cat_embeddings.append(embedding_layer(input_layer))
    
    def construir_modelo(self):
        # Entradas del modelo
        seq_mult_input = Input(shape=(self.timesteps, self.var_numericas + 1), name='seq_input')       # Ventas y variables numéricas históricas
        target_input = Input(shape=(self.var_numericas,), name='target_input')    # Variables numéricas del día objetivo
        self.create_categorical_modules()

        lstm_features = Concatenate(axis=-1)([seq_mult_input] + self.cat_embeddings)

        lstm_layer = LSTM(128,input_shape=(self.timesteps, lstm_features.shape[-1]), name='lstm_submodule')(lstm_features)
        
        dense_target = Dense(128, activation='relu', name='dense_target')(target_input)

        # Combinar salida de LSTM y variables del día objetivo
        merged = Concatenate(name='final_concat')([lstm_layer, dense_target])

        # Capa final para predicción
        dense_merged = Dense(64, activation='relu')(merged)
        output = Dense(1, activation='relu', name='output')(dense_merged)

        # Modelo completo
        model = Model(inputs=[seq_mult_input, target_input] + self.cat_inputs, outputs=output)
        return model

"""
var_categoricas = [
    {'name': 'family', 'num_categories': 34}, 
    {'name': 'store_nbr', 'num_categories': 55}
    ]
timesteps = 7
var_numericas = 4
data_manager = DataManager()
hybridModel = HybridModelBuilder(data_manager.generate_metadata())
modelo = hybridModel.construir_modelo()
modelo.summary()
x_train, y_train, x_val, y_val = data_manager.build_input_data()
optimizer = Adam(learning_rate=0.0001)
modelo.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
history = modelo.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=1024)
"""