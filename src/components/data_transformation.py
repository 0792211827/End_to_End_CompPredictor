import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, FunctionTransformer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
        # Label encoders for categorical columns
        self.label_encoders = {
            'Department': LabelEncoder(),
            'Job Title': LabelEncoder(),
            'Full/Part Time': LabelEncoder()
        }

    def get_data_transformer_object(self):
        '''
        Creates and returns the ColumnTransformer with preprocessing pipelines
        for both numerical and categorical columns.
        '''
        try:
            numerical_columns = ["Hourly Rate", "Period_worked", "Benefits Category"]
            categorical_columns = ["Department", "Job Title", "Full/Part Time"]

            # Pipeline for numerical features: imputing missing values and scaling
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Pipeline for categorical features: label encoding and scaling
            cat_pipeline = Pipeline(
                steps=[
                    ("label_encoder", FunctionTransformer(self.apply_label_encoding, validate=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # Combining both pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            logging.info(f"Preprocessor created with numerical columns: {numerical_columns} and categorical columns: {categorical_columns}")
            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def apply_label_encoding(self, X):
        '''
        Applies label encoding to specified categorical columns.
        '''
        for column, encoder in self.label_encoders.items():
            X[column] = encoder.fit_transform(X[column].astype(str))
        return X

    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function initiates the data transformation by applying pre-processing steps 
        and dropping unnecessary columns, transforming date columns, and generating target variables.
        '''
        try:
            # Read the dataset from the provided path
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed.")

            # Drop unnecessary columns
            train_df = train_df.drop(columns=["Unnamed: 11", "Name", "Termination Date"], errors='ignore')
            test_df = test_df.drop(columns=["Unnamed: 11", "Name", "Termination Date"], errors='ignore')
            logging.info("Dropped 'Unnamed: 11', 'Name', and 'Termination Date' columns.")

            # Drop missing values and duplicates
            train_df = train_df.dropna().drop_duplicates()
            test_df = test_df.dropna().drop_duplicates()
            logging.info("Dropped missing values and duplicates from train and test DataFrames.")




            # Convert 'Hire Date' to datetime format and calculate period worked
            train_df['Hire Date'] = pd.to_datetime(train_df['Hire Date'], errors='coerce')
            test_df['Hire Date'] = pd.to_datetime(test_df['Hire Date'], errors='coerce')
            latest_hire_date = train_df['Hire Date'].max()
            train_df['Period_worked'] = (latest_hire_date - train_df['Hire Date']).dt.days
            test_df['Period_worked'] = (latest_hire_date - test_df['Hire Date']).dt.days
            train_df = train_df.drop(columns=["Hire Date"], errors='ignore')
            test_df = test_df.drop(columns=["Hire Date"], errors='ignore')
            logging.info("Processed 'Hire Date' and calculated 'Period_worked'.")

            # Create 'Total_compensation' column as the sum of 'Regular Pay', 'Overtime Pay', and 'Other Pay'
            train_df['Total_compensation'] = train_df['Regular Pay'] + train_df['Overtime Pay'] + train_df['Other Pay']
            test_df['Total_compensation'] = test_df['Regular Pay'] + test_df['Overtime Pay'] + test_df['Other Pay']
            train_df = train_df.drop(columns=["Regular Pay", "Overtime Pay", "Other Pay"], errors='ignore')
            test_df = test_df.drop(columns=["Regular Pay", "Overtime Pay", "Other Pay"], errors='ignore')
            logging.info("Created 'Total_compensation' as target variable.")

            # Apply preprocessing
            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "Total_compensation"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combining transformed features with target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)




