import pandas as pd
from src.config import CONFIG

class DataLoader:
    def __init__(self, config=CONFIG):
        self.config = config
        self.index = config.get_variable('index')

    def load_train_data(self, n_rows=None):
        return pd.read_csv(self.config.get_path("train"), parse_dates=["date"], nrows=n_rows, index_col=self.index)

    def load_test_data(self, n_rows=None):
        return pd.read_csv(self.config.get_path("test"), parse_dates=["date"], nrows=n_rows, index_col=self.index)

    def load_store_metadata(self):
        return pd.read_csv(self.config.get_path("store_metadata"))
    
    def load_train_parquet(self, columns=None, n_rows=None):
        """
        Carga el archivo de entrenamiento en formato Parquet.
        Parámetros:
            columns (list, opcional): columnas a cargar del Parquet (para optimizar memoria)
        """
        df = pd.read_parquet(self.config.get_path("train_parquet"), columns=columns)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index(self.index).iloc[:n_rows] if n_rows else df.set_index(self.index)
    
    def load_test_parquet(self, columns=None, n_rows=None):
        """
        Carga el archivo de entrenamiento en formato Parquet.
        Parámetros:
            columns (list, opcional): columnas a cargar del Parquet (para optimizar memoria)
        """
        df = pd.read_parquet(self.config.get_path("test_parquet"), columns=columns)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index(self.index).iloc[:n_rows] if n_rows else df.set_index(self.index)
