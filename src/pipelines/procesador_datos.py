import yaml
import time
import pickle
import requests
import tensorflow as tf
from tqdm import tqdm
from pandas import read_csv, Categorical
from numpy import array, full, concatenate, newaxis
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class DataManager():
    def __init__(self):
        # Load data
        start_time = time.time()
        config = self._load_config()
        self.df_train = read_csv(config['paths']['train'], parse_dates=['date'], index_col='id')
        self.df_test = read_csv(config['paths']['test'], parse_dates=['date'], index_col='id')
        self.train_data_path = config['paths']['train_data']
        self.scaler_path = config['paths']['scaler']
        self.timesteps = config['timesteps']
        self.normalize = config['transformations']['normalize']
        self.date_columns = config['transformations']['date_columns']
        self.var_cat_st = config['variables']['categorical']['static']
        self.var_cat_dy = config['variables']['categorical']['dynamic']
        self.var_cat = self.var_cat_st + self.var_cat_dy
        self.target = config['variables']['target']
        self.categories = {}
        self.atributos = ['x_store','x_family', 'xy_historic', 'x_current','y_current']       
        self.X_train_raw = self._transform_df(self.df_train)
        end_time = time.time()
        print(f"Inicialización terminada: {end_time - start_time:.4f} segundos")
    
    def _load_config(self):
        with open('configs/project_config.yaml', "r") as f:
            return yaml.safe_load(f)
    
    def __get_n_var_num(self):
        return len(self.X_train_raw.columns) - len(self.var_cat) - 2

    def generate_metadata(self):
        meta_categories = {}
        for var in self.var_cat:
            n_cat = self.X_train_raw[var].nunique()
            meta_categories[var] = n_cat
        return {'categorical': meta_categories, 'var_num': self.__get_n_var_num()}
    
    def _embed_var_cat(self, df, test=False):
        for var in self.var_cat:
            if not test:
                df[var] = df[var].astype('category').cat.codes
                self.categories[var] = df[var].astype('category').cat.categories
            else:
                df[var] = Categorical(df[var], categories=self.categories[var]).codes
        return df
    
    def _add_date_columns(self, df):
        df['day'] = df.date.dt.day
        df['month'] = df.date.dt.month
        df['year'] = df.date.dt.year
        return df
    
    def _normalize_data(self, df):
        self.scaler_data = MinMaxScaler()
        scaler = MinMaxScaler()
        if self.normalize:
            df[self.target] = self.scaler_data.fit_transform(self.df_train[[self.target]])
        excluded_columns = [self.target, 'date'] + self.var_cat
        cols_norm = df.columns.difference(excluded_columns)
        df[cols_norm] = scaler.fit_transform(df[cols_norm])
        with open(self.scaler_path, "wb") as file:
            pickle.dump(self.scaler_data, file)
        return df
    
    def _transform_df(self, df):
        df = self._embed_var_cat(df)
        if self.date_columns:
            df = self._add_date_columns(df)
        df = self._normalize_data(df)
        return df
        
    
    def _create_dataset(self, store_nbr, family):
        filter_col = (self.df_train.store_nbr == store_nbr) & (self.df_train.family == family)
        return self.X_train_raw[filter_col].drop(columns=['store_nbr', 'family'])
    
    def _create_steps(self, df):
        df= df.drop(columns=['date'])
        xy_historic = array([df.iloc[i:i+self.timesteps].values for i in range(len(df) - self.timesteps)])
        x_current = df.drop(columns=[self.target]).iloc[self.timesteps:].values
        y_current = df[self.target].iloc[self.timesteps:].values
        return [xy_historic, x_current, y_current]
    
    def generate_X_train(self):
        train_data = {attr: [] for attr in self.atributos}
        for store in tqdm(self.X_train_raw.store_nbr.unique()):
            for family in self.X_train_raw.family.unique():
                df = self._create_dataset(store, family)
                n_dim = len(df) - self.timesteps
                new_values =  [full(n_dim, store), full(n_dim, family)] + self._create_steps(df)
                for attr, new_value in zip(self.atributos, new_values):
                    train_data[attr].append(new_value) 
        for attr in self.atributos:
            train_data[attr] = concatenate(train_data[attr], axis=0)
        
        with open(self.train_data_path, "wb") as file:
            pickle.dump(train_data, file)

        return train_data
    
    def __get_split_data(self):
        if self.load:
            train_data = self.generate_X_train()
        else:
            response = requests.get(self.train_data_path)
            train_data = pickle.loads(response.content)
        # Apply train_test_split to all attributes in `data_list`
        split_results = train_test_split(*train_data.values(), test_size=0.2, random_state=42)

        # Dynamically assign train and test variables
        for i, attr in enumerate(self.atributos):
            setattr(self, f"{attr}_train", split_results[2 * i])  # Train
            setattr(self, f"{attr}_test", split_results[2 * i + 1])  # Test

        # Return train and test data as dictionaries for easy access
        return {attr: getattr(self, f"{attr}_train") for attr in self.atributos}, \
            {attr: getattr(self, f"{attr}_test") for attr in self.atributos}
    
    def build_input_data(self, load=False):
        self.load = load
        data_train, data_test = self.__get_split_data()
        input_mapping = {
            "seq_input": "xy_historic",
            "target_input": "x_current",
            "store_nbr_input": "x_store",
            "family_input": "x_family"
        }
        datasets = {} 
        for split, dataset in zip(["train", "test"], [data_train, data_test]):
            datasets[f"x_{split}"] = {key: tf.convert_to_tensor(dataset[value], dtype=tf.int32) if key in ["store_nbr_input", "family_input"]
                                    else tf.convert_to_tensor(dataset[value], dtype=tf.float32)
                                    for key, value in input_mapping.items()}
            datasets[f"y_{split}"] = tf.convert_to_tensor(dataset["y_current"], dtype=tf.float32)
        return datasets["x_train"], datasets["y_train"], datasets["x_test"], datasets["y_test"]

    


        
