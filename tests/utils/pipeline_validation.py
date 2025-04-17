import pandas as pd
from src.pipelines.pipeline_builder import PipelineBuilder

def run_pipeline_validation(df, expected_cols=None, metadata_path=None, oil_path=None):
    expected_cols = expected_cols or [
        "store_nbr", "family", "sales", "onpromotion",
        "city", "state", "type", "cluster",
        "day_sin", "day_cos", "month_sin", "month_cos",
        "weekday_sin", "weekday_cos", "is_weekend"
    ]
    builder = PipelineBuilder()
    if metadata_path is None:
        pipeline = builder.build_preprocessor_pipeline()  # usa el default
    else:
        pipeline = builder.build_preprocessor_pipeline(metadata_path=metadata_path, oil_path=oil_path)
    transformed = pipeline.fit_transform(df)
    null_cols = transformed.columns[transformed.isnull().any()].tolist()
    assert isinstance(transformed, pd.DataFrame)
    assert not transformed.isnull().any().any(), f"Hay valores nulos en las columnas: {null_cols}"
    assert transformed.shape[0] == df.shape[0], "Se perdieron filas en el pipeline"

    for col in expected_cols:
        assert col in transformed.columns, f"Falta columna: {col}"

    for col in ["sales", "onpromotion"]:
        assert transformed[col].between(0, 1).all(), f"{col} no está correctamente normalizado"

    return transformed