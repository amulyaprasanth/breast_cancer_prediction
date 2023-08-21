import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts','raw_data.csv')
    train_data_path = os.path.join('artifacts','train_data.csv')
    test_data_path = os.path.join('artifacts','test_data.csv')
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion phase initiated.")
        
        try:
            data = pd.read_csv('data\data_cleaned.csv')
            logging.info("Read data in the form of DataFrame")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train test split initiated")
            
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data Ingestion  is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()