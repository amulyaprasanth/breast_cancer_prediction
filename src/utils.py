import os
import sys
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
            
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(filepath):
    try:
        with open(filepath, 'rb') as f:
           return pickle.load(f)
       
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(params.keys())[i]]
            
            # hyperparameter tuning
            gs_model = GridSearchCV(estimator=model, param_grid=param, cv=3, refit=True,
                                    n_jobs=-1, verbose=False)
            gs_model.fit(X_train, y_train)

            model.set_params(**gs_model.best_params_)

            # fitting the best model
            model.fit(X_train, y_train)

            # making predictions
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            
            # creating model report
            # for training data
            train_acc = accuracy_score(y_train, train_preds)
            train_prec = precision_score(y_train, train_preds)
            train_recall = recall_score(y_train, train_preds)
            train_f1 = f1_score(y_train, train_preds)
            
            # for testing data
            test_acc = accuracy_score(y_test, test_preds)
            test_prec = precision_score(y_test, test_preds)
            test_recall = recall_score(y_test, test_preds)
            test_f1 = f1_score(y_test, test_preds)
            
            report[list(models.keys())[i]] = test_acc
            
        return report

    
    except Exception as e:
        raise CustomException(e, sys)