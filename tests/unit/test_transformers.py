import pandas as pd
from src.features import DatePartAdder, CategoricalEncoder, AdaptiveScaler, StoreMetadataMerger

# Test para DatePartAdder
def test_datepart_adder_creates_columns(sample_date_df):
    transformed = DatePartAdder().fit_transform(sample_date_df)
    assert "day" in transformed.columns
    assert "month" in transformed.columns
    assert "weekday" in transformed.columns
    assert transformed["day"].tolist() == [1, 15]
    assert transformed["month"].tolist() == [1, 6]

# Test para CategoricalEncoder
def test_categorical_encoder_encodes_correctly(sample_categorical_df):
    encoder = CategoricalEncoder(columns=["family"])
    transformed = encoder.fit_transform(sample_categorical_df)
    assert "family" in transformed.columns
    assert transformed["family"].nunique() == 3
    assert transformed["family"].isin(range(3)).all()

# Test para AdaptiveScaler
def test_adaptive_scaler_scales_between_0_and_1(sample_numeric_df):
    scaler = AdaptiveScaler(excluded_columns=["store_nbr", "family"])
    scaled = scaler.fit_transform(sample_numeric_df)
    for col in ["sales", "onpromotion"]:
        assert col in scaled.columns
        assert scaled[col].min() >= 0
        assert scaled[col].max() <= 1

# Test para StoreMetadataMerger
def test_store_metadata_merger_adds_columns(sample_store_df):
    df = pd.DataFrame({"store_nbr": [1, 2]})
    merger = StoreMetadataMerger(metadata_path=str(sample_store_df))
    merged = merger.fit_transform(df)

    assert "city" in merged.columns
    assert "type" in merged.columns
    assert merged.shape[0] == df.shape[0]