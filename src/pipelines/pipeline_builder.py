from sklearn.pipeline import Pipeline
from src.features import DatePartAdder, StoreMetadataMerger, CategoricalEncoder, CyclicEncoder, WeekendFlagger, AdaptiveScaler 
from src.config import CONFIG

def build_pipeline(metadata_path=CONFIG.get_path("store_metadata")):
    return Pipeline([
        # 1. Añadir metadata de tienda (store_nbr → ciudad, tipo, etc.)
        ("merge_metadata", StoreMetadataMerger(metadata_path=metadata_path)),

        # 2. Extraer partes de la fecha (day, month, weekday)
        ("date_parts", DatePartAdder()),

        # 3. Codificar categorías a enteros
        ("categorical", CategoricalEncoder(columns=[
            "store_nbr", "family", "type", "city", "state"
        ])),

        # 4. Agregar flag binario para fin de semana
        ("weekend_flag", WeekendFlagger()),

        # 5. Codificación cíclica para variables temporales
        ("cyclic", CyclicEncoder(columns_config={
            "day": 31,
            "month": 12,
            "weekday": 7
        })),

        # 6. Escalar numéricos de forma adaptable
        ("scaling", AdaptiveScaler(columns=[
            "onpromotion", "sales", "day_sin", "day_cos", "month_sin", "month_cos",
            "weekday_sin", "weekday_cos", "is_weekend"
        ])),
    ])