import os 
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransforamtionConfig:
    preprocessor_obj_path = os.path.join('artifacts', 'preprocessor_obj.pkl')
    
class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransforamtionConfig()
        
    def get_preprocessor(self):
        try:
            num_features = ['smoothness_mean',
                                'symmetry_mean',
                                'fractal_dimension_mean',
                                'texture_se',
                                'smoothness_se',
                                'compactness_se',
                                'concavity_se',
                                'concave points_se',
                                'symmetry_se',
                                'fractal_dimension_se',
                                'smoothness_worst',
                                'symmetry_worst',
                                'fractal_dimension_worst']
            
            # Initiliaze the transformers
            num_transformer = StandardScaler()            
            
            # Create the transformation pipeline
            preprocessor = ColumnTransformer(
                [
                    ("num_transformer", num_transformer, num_features)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Reading training and testing data")
            
            logging.info("Getting preprocessor object")
            
            preprocessor_obj = self.get_preprocessor()
            label_enc = LabelEncoder()
            
            target_column = "diagnosis"
                        
            input_train_features_df = train_df.drop(target_column, axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_test_features_df = test_df.drop(target_column, axis=1)
            target_feature_test_df = test_df[target_column]
            
            logging.info("Applying preprocessing object to training and testing dataframe")
            
            input_features_train_arr = preprocessor_obj.fit_transform(input_train_features_df)
            input_feature_test_arr = preprocessor_obj.transform(input_test_features_df)
            
            target_feature_train_arr = label_enc.fit_transform(target_feature_train_df)
            target_feature_test_arr = label_enc.transform(target_feature_test_df)
            
            train_arr = np.c_[input_features_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_features_train_arr, target_feature_test_arr]
            
            logging.info("Applying preprocessing object to training and testing data")
            
            logging.info("Saving preprocessor object")
            save_object(self.data_transformation_config.preprocessor_obj_path, preprocessor_obj)

            return (
                self.data_transformation_config.preprocessor_obj_path,
                train_arr,
                test_arr
            )
        except Exception as e:
            raise CustomException(e, sys)