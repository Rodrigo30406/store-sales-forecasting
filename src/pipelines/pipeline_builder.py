from sklearn.pipeline import Pipeline
from src.features import DatePartAdder, CategoricalEncoder, CyclicEncoder, WeekendFlagger, AdaptiveScaler

def build_pipeline():
    return Pipeline([
        ("date_parts", DatePartAdder()),
        ("categorical", CategoricalEncoder(columns=["store_nbr", "family"])),
        ("cyclic", CyclicEncoder(columns_config={"day": 31, "month": 12, "weekday": 7})),
        ("weekend_flag", WeekendFlagger()),
        ("scaling", AdaptiveScaler(columns=[
            "onpromotion", "sales", "is_weekend",
            "day_sin", "day_cos", "month_sin", "month_cos", "weekday_sin", "weekday_cos"
        ]))
    ])