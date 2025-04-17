from sklearn.pipeline import Pipeline
from src.features import DatePartAdder, DataMerger, CategoricalEncoder, CyclicEncoder, WeekendFlagger, AdaptiveScaler 
from lightgbm import LGBMRegressor
from src.config import CONFIG

class PipelineBuilder:
    def __init__(self, config=CONFIG):
        self.cat_columns = config.get_variable('categorical', flatten=True)
        self.model_params = config.get_model_params()
        self.preprocessor = self.build_preprocessor_pipeline()

    def build_preprocessor_pipeline(self, metadata_path=CONFIG.get_path('store_metadata'), oil_path=CONFIG.get_path('oil_path')):
        return Pipeline([
            # 1. Añadir metadata de tienda (store_nbr → ciudad, tipo, etc.)
            ("merge_metadata", DataMerger(metadata_path=metadata_path, on='store_nbr')),

            # 1. Añadir data de precios del petroleo (date → oil_price)
            ("merge_oil_price", DataMerger(metadata_path=oil_path, on='date')),

            # 2. Extraer partes de la fecha (day, month, weekday)
            ("date_parts", DatePartAdder()),

            # 3. Codificar categorías a enteros
            ("categorical", CategoricalEncoder(columns=self.cat_columns)),

            # 4. Agregar flag binario para fin de semana
            ("weekend_flag", WeekendFlagger()),

            # 5. Codificación cíclica para variables temporales
            ("cyclic", CyclicEncoder()),

            # 6. Escalar numéricos de forma adaptable
            ("scaling", AdaptiveScaler(excluded_columns=self.cat_columns)),
        ])

    def build_lgbm_pipeline(self, model_params=None):
        if model_params is None:
            model_params = self.model_params

        return Pipeline([
            ("preprocessor", self.preprocessor),
            ("model", LGBMRegressor(**model_params))
        ])