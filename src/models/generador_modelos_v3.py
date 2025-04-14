import math
import yaml
from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Concatenate, Flatten, RepeatVector, Dropout, BatchNormalization # type: ignore


class HybridModel(Model):
    def __init__(self, metadata):
        """
        Inicializa el modelo híbrido con las entradas y salidas ya configuradas, de manera que el modelo
        esté listo para ser utilizado directamente (como ResNet50).

        Args:
            categorical_features (list): Lista de variables categóricas con nombres y cantidad de categorías.
            timesteps (int): Cantidad de pasos temporales en las secuencias.
            var_numericas (int): Número de variables numéricas por muestra.
        """
        super(HybridModel, self).__init__()
        
        self.categorical_features = metadata['categorical']
        self.timesteps = self._load_config()['timesteps']
        self.var_numericas = metadata['var_num']

        # Crear módulos de embedding para variables categóricas
        self.embedding_layers = []
        self.repeat_layers = []
        self.embedding_dense_layers = []
        for key in self.categorical_features.keys():
            output_dim = self.calcular_output_dim(self.categorical_features[key])
            embedding_layer = Sequential([
                Embedding(input_dim=self.categorical_features[key] + 1, output_dim=output_dim, name=f"{key}_embed"),
                Flatten(),
            ], name=f"{key}_embed_module")
            self.embedding_layers.append(embedding_layer)

            # Repeat vector para LSTM
            self.repeat_layers.append(RepeatVector(self.timesteps, name=f"{key}_repeat"))

            # Dense para información categórica directa
            self.embedding_dense_layers.append(Dense(128, activation='relu', name=f"{key}_dense"))

        # Definir las capas principales del modelo
        self.lstm_layer = LSTM(128, name='lstm_submodule')
        self.bn_lstm = BatchNormalization()
        self.dp_lstm = Dropout(0.3)
        self.dense_target = Dense(128, activation='relu', name='dense_target')
        self.dense_merged_target = Dense(64, activation='relu', name='dense_target')
        self.bn_dense_target = BatchNormalization()
        self.dp_target = Dropout(0.3)
        self.final_concat = Concatenate(name='final_concat')
        self.dense_merged = Dense(32, activation='relu')
        self.bn_dense_merged = BatchNormalization()
        self.dp_merged = Dropout(0.5)
        self.output_layer = Dense(1, activation='relu', name='output')

        # Construir las entradas del modelo
        self.inputs = self._create_inputs()
        self.outputs = self.call(self.inputs)  # Generar las salidas desde el forward pass

    def _load_config(self):
        with open('configs/project_config.yaml', "r") as f:
            return yaml.safe_load(f)
        
    def calcular_output_dim(self, input_dim, d=2):
        """
        Calcula la dimensión de salida para la capa de embedding.
        """
        return math.ceil(d * math.log(input_dim))

    def _create_inputs(self):
        """
        Crea las entradas del modelo funcional, incluyendo secuencias históricas y variables categóricas.
        """
        inputs = {
            'seq_input': Input(shape=(self.timesteps, self.var_numericas + 1), name='seq_input'),
            'target_input': Input(shape=(self.var_numericas,), name='target_input'),
        }
        for key in self.categorical_features.keys():
            inputs[f"{key}_input"] = Input(shape=(1,), name=f"{key}_input")
        return inputs

    def call(self, inputs):
        """
        Define el forward pass del modelo híbrido.

        Args:
            inputs (list): Lista de tensores de entrada.

        Returns:
            Tensor: Salida del modelo.
        """
        seq_mult_input = inputs['seq_input']
        target_input = inputs['target_input']
        cat_inputs = [inputs[f"{key}_input"] for key in self.categorical_features.keys()]

        # Procesar embeddings categóricos
        cat_embeddings = [layer(cat_input) for layer, cat_input in zip(self.embedding_layers, cat_inputs)]

        cat_emb_repeated = [repeat(e) for repeat, e in zip(self.repeat_layers, cat_embeddings)]

        # Combinar características numéricas históricas y embeddings
        lstm_features = Concatenate(axis=-1)([seq_mult_input] + cat_emb_repeated)
        lstm_out = self.lstm_layer(lstm_features)
        batch_lstm = self.bn_lstm(lstm_out)
        dp_lstm_out = self.dp_lstm(batch_lstm)

        cat_emb_dense = [dense(e) for dense, e in zip(self.embedding_dense_layers, cat_embeddings)]

        # Procesar entrada numérica del día objetivo
        dense_target_out = self.dense_target(target_input)
        cat_info_flat = Concatenate(axis=-1)(cat_emb_dense + [dense_target_out])
        merged_target_dense = self.dense_merged_target(cat_info_flat)
        batch_target = self.bn_dense_target(merged_target_dense)
        dp_target_out = self.dp_target(batch_target)


        # Combinar salidas
        merged_features = self.final_concat([dp_lstm_out, dp_target_out])
        dense_out = self.dense_merged(merged_features)
        batch_out = self.bn_dense_merged(dense_out)
        dp_out = self.dp_merged(batch_out)
        return self.output_layer(dp_out)
