import numpy as np
from yaml import safe_load
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
from src.features.outliers_removal import OutliersRemover
import joblib
import sys
import logging

COLUMN_NAMES = ['pickup_latitude',
                'pickup_longitude',
                'dropoff_latitude',
                'dropoff_longitude']

TARGET = 'trip_duration'

def save_transformer(path,object):
    joblib.dump(value=object,
                filename=path)

def remove_outliers(dataframe:pd.DataFrame, percentiles:list, column_names:list) -> pd.DataFrame:
    df = dataframe.copy()
    
    outlier_transformer = OutliersRemover(percentile_values=percentiles,
                                          col_subset=column_names)
    
    # fit on the data
    outlier_transformer.fit(dataframe)
     
    return outlier_transformer

#* Vendor -id OHE, date columns
#* lat/long - min max scale
#* distances - Standard scale

def train_preprocessor(data:pd.DataFrame):
    ohe_columns = ['vendor_id']
    standard_scale_columns = ['haversine_distance', 'euclidean_distance',
       'manhattan_distance']
    min_max_scale_columns = ['pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
    
    preprocessor = ColumnTransformer(transformers=[
        ('one-hot',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'),ohe_columns),
        ('min-max',MinMaxScaler(),min_max_scale_columns),
        ('standard-scale',StandardScaler(),standard_scale_columns)
    ],remainder='passthrough',verbose_feature_names_out=False,n_jobs=1)
    
    # set the output as df
    preprocessor.set_output(transform='pandas')
    # fit the preprocessor on training data
    preprocessor.fit(data)
    
    return preprocessor

def transform_data(transformer,data:pd.DataFrame):
    
    # transform the data
    data_transformed = transformer.transform(data)
    
    return data_transformed

def transform_output(target:pd.Series):
    target_values = np.asarray(target).reshape(-1, 1)
    if not np.isfinite(target_values).all():
        raise ValueError('Target contains non-finite values; cannot apply natural log.')
    if (target_values <= 0).any():
        raise ValueError('All trip_duration target values must be > 0 for np.log.')

    log_transformer = FunctionTransformer(
        func=np.log,
        inverse_func=np.exp,
        validate=False,
        check_inverse=False,
    )
    log_transformer.fit(target_values)
    return log_transformer

def read_dataframe(path):
    df = pd.read_csv(path)
    return df

def save_dataframe(dataframe:pd.DataFrame, save_path):
    dataframe.to_csv(save_path,index=False)

    
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info('Starting data preprocessing.')
    # current file path
    current_path = Path(__file__)
    # root directory path
    root_path = current_path.parent.parent.parent
    # input_data path
    input_path = root_path / 'data' / 'processed' / 'build-features'
    # read from the parameters file
    with open('params.yaml') as f:
        params = safe_load(f)
    # percentile values
    percentiles = list(params['data_preprocessing']['percentiles'])
    logger.info('Loaded preprocessing percentiles: %s', percentiles)
    # save transformers path
    save_transformers_path = root_path / 'models' / 'transformers'
    # make directory
    save_transformers_path.mkdir(parents=True, exist_ok=True)
    # save output file path
    save_data_path = root_path / 'data' / 'processed' / 'final'
    # make directory
    save_data_path.mkdir(exist_ok=True)
    
    for filename in sys.argv[1:]:
        logger.info('Processing file: %s', filename)
        complete_input_path = input_path / filename
        if filename == 'train.csv':
            df = read_dataframe(complete_input_path)
            logger.info('Loaded training data: %d rows, %d columns.', len(df), len(df.columns))
            # make X and y
            X = df.drop(columns=TARGET)
            y = df[TARGET]
            logger.info('Fitting outlier transformer on training features.')
            # remove outliers from data
            outlier_transformer = remove_outliers(dataframe=X,percentiles=percentiles,
                                                  column_names=COLUMN_NAMES)
            # save the transformer
            save_transformer(path=save_transformers_path / 'outliers.joblib',
                             object=outlier_transformer)
            # transform the data
            df_without_outliers = transform_data(transformer=outlier_transformer,
                                                 data=X)                
            logger.info('Fitting feature preprocessor on training features.')
            # train the preprocessor on the data
            preprocessor = train_preprocessor(data=df_without_outliers)
            # save the preprocessor
            save_transformer(path=save_transformers_path / 'preprocessor.joblib',
                             object=preprocessor)
            # transform the data
            X_trans = transform_data(transformer=preprocessor,
                                     data=df_without_outliers)
            # fit the natural-log target transformer on training targets
            y_filtered = y.loc[df_without_outliers.index]
            logger.info('Fitting natural-log target transformer on %d training targets.', len(y_filtered))
            output_transformer = transform_output(y_filtered)
            logger.info('Natural-log target transformer fitted.')
            # transform the target
            y_trans = transform_data(transformer=output_transformer,
                                     data=y_filtered.values.reshape(-1,1))
            # save the transformed output to the df
            X_trans['trip_duration'] = y_trans
            # save the output transformer
            save_transformer(path=save_transformers_path / 'output_transformer.joblib',
                             object=output_transformer)
            logger.info('Saved target transformer to %s.', save_transformers_path / 'output_transformer.joblib')
            
            # save the transformed data
            save_dataframe(dataframe=X_trans,
                           save_path=save_data_path / filename)
            logger.info('Saved processed data to %s.', save_data_path / filename)
            
        elif filename == 'val.csv':
            df = read_dataframe(complete_input_path)
            logger.info('Loaded validation data: %d rows, %d columns.', len(df), len(df.columns))
            # make X and y
            X = df.drop(columns=TARGET)
            y = df[TARGET]
            # load the transfomer
            outlier_transformer = joblib.load(save_transformers_path / 'outliers.joblib')
            df_without_outliers = transform_data(transformer=outlier_transformer,
                                                data=X)                
            # load the preprocessor
            preprocessor = joblib.load(save_transformers_path / 'preprocessor.joblib')
            # transform the data
            X_trans = transform_data(transformer=preprocessor,
                                    data=df_without_outliers)
            y_filtered = y.loc[df_without_outliers.index]
            # load the training-fitted natural-log target transformer
            output_transformer = joblib.load(save_transformers_path / 'output_transformer.joblib') 
            logger.info('Applying training-fitted natural-log transformer to validation target.')
            # transform the target
            y_trans = transform_data(transformer=output_transformer,
                                    data=y_filtered.values.reshape(-1,1))
            # save the transformed output to the df
            X_trans['trip_duration'] = y_trans
            
            # save the transformed data
            save_dataframe(dataframe=X_trans,
                        save_path=save_data_path / filename)
            
        elif filename == 'test.csv':
            df = read_dataframe(complete_input_path)
            logger.info('Loaded test data: %d rows, %d columns.', len(df), len(df.columns))
            # load the transfomer
            outlier_transformer = joblib.load(save_transformers_path / 'outliers.joblib')
            df_without_outliers = transform_data(transformer=outlier_transformer,
                                                data=df)                
            # load the preprocessor
            preprocessor = joblib.load(save_transformers_path / 'preprocessor.joblib')
            # transform the data
            X_trans = transform_data(transformer=preprocessor,
                                    data=df_without_outliers)
            # save the transformed data
            save_dataframe(dataframe=X_trans,
                        save_path=save_data_path / filename)
            
if __name__ == "__main__":
    main()