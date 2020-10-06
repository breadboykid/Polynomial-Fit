import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

class Polynomialfit():
    
    def __init__(self, data_frame, low_degree, high_degree, x_curve_start, x_curve_stop, linspace_num=50, test_size = 0.2, random_state = None):
        self.data_frame = data_frame
        self.low_degree = low_degree
        self.high_degree = high_degree
        self.dataFrameHandler(self.data_frame)
        self.stratifiedHandler(test_size, random_state)
        self.polynomialEstimatorHandler()
        self.X_curve = self.getXCurve(x_curve_start, x_curve_stop, linspace_num)
        self.y_curve = self.getYCurve()
        
    def dataFrameHandler(self, data_frame):
        self.X = data_frame.values[:,0].reshape(-1,1)
        self.y = data_frame.values[:,1]
        
    def stratifiedHandler(self, test_sizing, rdm_state):
        bins = np.round(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size = test_sizing, stratify = bins, random_state=rdm_state)
        
    def polynomialEstimatorHandler(self):
        pipeline = Pipeline((
            ("poly_features", PolynomialFeatures(degree=2)),
            ("lin_reg", LinearRegression(fit_intercept=False)),))

        degrees = range(1,20)
        parameters = {"poly_features__degree": degrees}

        grid_search = GridSearchCV(pipeline, parameters, cv=5)

        # Fit GridSearchCV object
        grid_search.fit(self.X_train, self.y_train)
        
        # extract the selected model
        self.best_model = grid_search.best_estimator_
        
    def getXCurve(self, x_curve_start, x_curve_stop, linspace_num):
        return np.linspace(x_curve_start, x_curve_stop, linspace_num).reshape(-1, 1)
    
    def getYCurve(self):
        return self.best_model.predict(self.X_curve)
            
    def plot(self):
        plt.title('Polynomial fit')
        plt.xlabel('Feature')
        plt.ylabel('Target value')        
        plt.plot(self.X_train, self.y_train, 'o', label='Training Data')
        plt.plot(self.X_test, self.y_test, 'o', label='Test Data')
        plt.plot(self.X_curve, self.y_curve, 'r')
        plt.legend()
        plt.show()