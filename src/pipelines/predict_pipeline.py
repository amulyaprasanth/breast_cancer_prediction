import sys
import pandas as pd


from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

from src.components.data_transformation import DataTransforamtionConfig
from src.components.model_trainer import ModelTrainerConfig


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:

            model_path = ModelTrainerConfig.trained_model_path
            preprocessor_obj_path = DataTransforamtionConfig().preprocessor_obj_path

            print("Loading model and preprocessor")
            model = load_object(model_path)
            preprocessor_obj = load_object(preprocessor_obj_path)
            print("Loading completed")

            print("Generating predictions...")
            data_scaled = preprocessor_obj.transform(features)
            prediction = model.predict(data_scaled)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 smoothness_mean,
                 symmetry_mean,
                 fractal_dimension_mean,
                 texture_se,
                 smoothness_se,
                 compactness_se,
                 concavity_se,
                 concave_points_se,
                 symmetry_se,
                 fractal_dimension_se,
                 smoothness_worst,
                 symmetry_worst,
                 fractal_dimension_worst):
        self.smoothness_mean = smoothness_mean
        self.symmetry_mean = symmetry_mean
        self.fractal_dimension_mean = fractal_dimension_mean
        self.texture_se = texture_se
        self.smoothness_se = smoothness_se
        self.compactness_se = compactness_se
        self.concavity_se = concavity_se
        self.concave_points_se = concave_points_se
        self.symmetry_se = symmetry_se
        self.fractal_dimension_se = fractal_dimension_se
        self.smoothness_worst = smoothness_worst
        self.symmetry_worst = symmetry_worst
        self.fractal_dimension_worst = fractal_dimension_worst
        
    def get_data_as_dataframe(self):
      try:
        custom_data_dict = {
          'smoothness_mean' : [self.smoothness_mean],
                                'symmetry_mean' : [self.symmetry_mean],
                                'fractal_dimension_mean' : [self.fractal_dimension_mean],
                                'texture_se' : [self.texture_se],
                                'smoothness_se' : [self.smoothness_se],
                                'compactness_se' : [self.compactness_se],
                                'concavity_se' : [self.concavity_se],
                                'concave points_se' : [self.concave_points_se],
                                'symmetry_se' : [self.symmetry_se],
                                'fractal_dimension_se' : [self.fractal_dimension_se],
                                'smoothness_worst' : [self.smoothness_worst],
                                'symmetry_worst' : [self.symmetry_worst],
                                'fractal_dimension_worst' : [self.fractal_dimension_worst], 
        }
        
        return pd.DataFrame(custom_data_dict)
      except Exception as e:
        raise CustomException(e, sys)
