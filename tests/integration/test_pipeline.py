from tests.utils import run_pipeline_validation

def test_build_pipeline_with_metadata_fixture(sample_train_df, sample_store_df, sample_oil_df):
    run_pipeline_validation(sample_train_df, metadata_path=str(sample_store_df), oil_path=str(sample_oil_df))

def test_pipeline_with_csv_data(data_loader):
    df = data_loader.load_train_data()
    print("Antes de entrar:",df.date.dtype)
    run_pipeline_validation(df)

def test_pipeline_with_parquet_data(data_loader):
    df = data_loader.load_train_parquet()
    run_pipeline_validation(df)