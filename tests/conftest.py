import pytest
import pandas as pd
import os
from src.ingestion.load import DataLoader

@pytest.fixture
def data_loader():
    return DataLoader()

@pytest.fixture
def sample_date_df():
    return pd.DataFrame({
        "date": ["2023-01-01", "2023-06-15"]
    })

@pytest.fixture
def sample_categorical_df():
    return pd.DataFrame({
        "family": ["A", "B", "A", "C"]
    })

@pytest.fixture
def sample_numeric_df():
    return pd.DataFrame({
        "sales": [100, 200, 300],
        "onpromotion": [1, 0, 1]
    })

@pytest.fixture
def sample_store_df(tmp_path_factory):
    df = pd.DataFrame({
        "store_nbr": [1, 2],
        "city": ["Lima", "Cusco"],
        "type": ["A", "B"],
        "state": ["Costa", "Sierra"],
        "cluster": ["1", "19"]
    })
    tmp_dir = tmp_path_factory.mktemp("data")
    file_path = tmp_dir / "temp_stores.csv"
    df.to_csv(file_path, index=False)
    yield file_path
    os.remove(file_path)  # Borrado del temporal al finalizar

@pytest.fixture
def sample_train_df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
        "store_nbr": [1, 2],
        "family": ["A", "B"],
        "onpromotion": [1, 0],
        "sales": [100.0, 200.0]
    })

@pytest.fixture
def sample_oil_df(tmp_path_factory):
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        "dcoilwtico": [70.5, 71.2, 69.8]
    })
    tmp_dir = tmp_path_factory.mktemp("data")
    file_path = tmp_dir / "temp_oil.csv"
    df.to_csv(file_path, index=False)
    yield file_path
    os.remove(file_path)