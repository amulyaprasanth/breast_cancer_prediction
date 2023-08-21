import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, evaluate_models, save_object

import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):

        try:
            logging.info("Split training and test input data")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                "Logistic Regression": LogisticRegression(),
                "SVC": SVC(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(),
                "CatBoost": CatBoostClassifier(verbose=False)
            }

            hyperparameters = {
                "Logistic Regression": {
                    "penalty": ["l1", "l2"],
                    "C": [0.1, 1, 10]
                },
                "SVC": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"]
                },
                "K-Neighbors Classifier": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"]
                },
                "Naive Bayes": {},
                "Decision Tree": {
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5, 10]
                },
                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5, 10]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 150],
                    "learning_rate": [0.1, 0.5, 1]
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100, 150],
                    "learning_rate": [0.1, 0.5, 1],
                    "max_depth": [3, 5, 7]
                },
                "XGBoost": {
                    "n_estimators": [50, 100, 150],
                    "learning_rate": [0.1, 0.5, 1],
                    "max_depth": [3, 5, 7]
                },
                "CatBoost": {
                    "iterations": [50, 100, 150],
                    "learning_rate": [0.1, 0.5, 1]
                }
            }

            logging.info("Training models")
            model_report = evaluate_models(X_train=X_train, y_train=y_train,
                                           X_test=X_test, y_test=y_test, models=models, params=hyperparameters)

            # get the best model
            logging.info("Getting the best model")
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info(
                f"Best found model on both training and testing dataset: {best_model}")

            save_object(
                filepath=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            acc = accuracy_score(y_test, predicted)
            return acc

        except Exception as e:
            raise CustomException(e, sys)
