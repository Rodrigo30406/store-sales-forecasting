import pandas as pd
from src.pipelines.pipeline_builder import build_pipeline
from tests.utils.validation import run_pipeline_validation
from src.ingestion.load import DataLoader

def test_build_pipeline_with_metadata_fixture(sample_train_df, sample_store_df):
    run_pipeline_validation(sample_train_df, metadata_path=str(sample_store_df))

def test_pipeline_with_csv_data(data_loader):
    df = data_loader.load_train_data()
    run_pipeline_validation(df)

def test_pipeline_with_parquet_data(data_loader):
    df = data_loader.load_train_parquet()
    run_pipeline_validation(df)