import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector


def preprocess_data(data_file_path):
   """Preprocesses the given dataset.

   Args:
       data_file_path (str): Path to the CSV file containing the dataset.

   Returns:
       pd.DataFrame: Preprocessed dataset.
   """

   df = pd.read_csv(data_file_path)
    FILE_PATH_Train = os.path.join(os.getcwd(), 'train.csv')
    df_train = pd.read_csv(FILE_PATH_Train)
    df_train = df_train.drop('Id', axis=1)


   # Add a new feature
   df['AvBedroomArea'] = df['GrLivArea'] / df['BedroomAbvGr']
   df_train['AvBedroomArea'] = df_train['GrLivArea'] / df_train['BedroomAbvGr']

   # Handle missing values and infinite values
   df = df.replace('missing', np.nan)
   df = df.replace([np.inf, -np.inf], np.nan)
    ## Split the whole Dataset to Feature & Target
    X_train = df_train.drop(columns=['SalePrice'], axis=1)   ## Features
    y_train = df_train['SalePrice']   ## target

   # Separate numerical and categorical columns
    num_cols = [col for col in X_train.columns if X_train[col].dtype in ['float32', 'float64', 'int32', 'int64']]
    categ_cols = [col for col in X_train.columns if X_train[col].dtype not in ['float32', 'float64', 'int32', 'int64']]
    num_cols = [col for col in df.columns if df[col].dtype in ['float32', 'float64', 'int32', 'int64']]
    categ_cols = [col for col in df.columns if df[col].dtype not in ['float32', 'float64', 'int32', 'int64']]
   # Create pipelines for numerical and categorical features
    ## We can get much much easier like the following
    # Replace 'missing' with a more appropriate value or NaN
    X_train = X_train.replace('missing', np.nan)
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    
    ## numerical pipeline
    num_pipeline = Pipeline([
                            ('selector', DataFrameSelector(num_cols)),    ## select only these columns
                            ('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler())
                            ])
    
    ## categorical pipeline
    categ_pipeline = Pipeline(steps=[
                ('selector', DataFrameSelector(categ_cols)),    ## select only these columns
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('OHE', OneHotEncoder(handle_unknown='ignore', sparse=False))])
    
    ## concatenate both two pipelines
    total_pipeline = FeatureUnion(transformer_list=[
                                                ('num_pipe', num_pipeline),
                                                ('categ_pipe', categ_pipeline)
                                                   ]
                                 )
    ## deal with (total_pipeline) as an instance -- fit and transform to train dataset and transform only to other datasets
    X_train_final = total_pipeline.fit_transform(X_train)
    # X_test_final = total_pipeline.transform(X_test)                 ### Every thing is processed :D

   # Apply preprocessing pipeline
   preprocessed_data = total_pipeline.transform(df)

   return preprocessed_data


def preprocess_new_data(new_data, feature_names):
   """Preprocesses new data instances.

   Args:
       new_data (np.ndarray): New data instances to be preprocessed.
       feature_names (list): Names of features in the same order as in new_data.

   Returns:
       np.ndarray: Preprocessed new data instances.
   """

   # Ensure new data is in a DataFrame with appropriate column names
   new_data_df = pd.DataFrame(new_data, columns=feature_names)

   # Apply the existing preprocessing pipeline
   preprocessed_new_data = full_pipeline.transform(new_data_df)

   return preprocessed_new_data

