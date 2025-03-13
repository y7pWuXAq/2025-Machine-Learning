from model_class.KNN_Model import *
from model_class.Linear_Model import *
from model_class.Ridge_Model import *
from model_class.Lasso_Model import *
from model_class.RandomForest_Model import *
from model_class.ExtraTrees_Model import *
from model_class.GradientBoosting_Model import *
from model_class.HistGradientBoosting_Model import *
from model_class.XGBRegressor_Model import *

def access_knn_model():
    # KNN_Model.py에 정의된 모든 클래스와 함수에 접근 가능
    model = KNNModel()  # LinearModel이 정의되어 있다고 가정
    return model

def access_linear_model():
    # Linear_Model.py에 정의된 모든 클래스와 함수에 접근 가능
    model = LinearModel()  # LinearModel이 정의되어 있다고 가정
    return model

def access_ridge_model():
    # Ridge_Model.py에 정의된 모든 클래스와 함수에 접근 가능
    model = RidgeModel()
    return model

def access_lasso_model():
    # Lasso_Model.py에 정의된 모든 클래스와 함수에 접근 가능
    model = LassoModel()
    return model

def access_randomforest_model():
    # RandomForest_Model.py에 정의된 모든 클래스와 함수에 접근 가능
    model = RandomForestModel()
    return model

def access_extratrees_model():
    # ExtraTrees_Model.py에 정의된 모든 클래스와 함수에 접근 가능
    model = ExtraTreesModel()
    return model

def access_gradient_model():
    # GradientBoosting_Model.py에 정의된 모든 클래스와 함수에 접근 가능
    model = GradientModel()
    return model

def access_histgradient_model():
    # GradientBoosting_Model.py에 정의된 모든 클래스와 함수에 접근 가능
    model = HistGradientModel()
    return model

def access_XGB_model():
    # XGBRegressor_Model.py에 정의된 모든 클래스와 함수에 접근 가능
    model = XGBModel()
    return model