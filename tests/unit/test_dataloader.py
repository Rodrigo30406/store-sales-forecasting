import pandas as pd
from src.ingestion.load import DataLoader
from src.config import CONFIG

def test_load_train_data(data_loader):
    df = data_loader.load_train_data(n_rows=5)
    assert isinstance(df, pd.DataFrame)
    assert "date" in df.columns
    assert df.shape[0] <= 5

def test_load_test_data(data_loader):
    df = data_loader.load_test_data(n_rows=5)
    assert isinstance(df, pd.DataFrame)
    assert "date" in df.columns
    assert df.shape[0] <= 5

def test_load_store_metadata(data_loader):
    df = data_loader.load_store_metadata()
    assert isinstance(df, pd.DataFrame)
    assert "store_nbr" in df.columns

def test_load_train_parquet(data_loader):
    df = data_loader.load_train_parquet()
    assert isinstance(df, pd.DataFrame)
    assert "store_nbr" in df.columns