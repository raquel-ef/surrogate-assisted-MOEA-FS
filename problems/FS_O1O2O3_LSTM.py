# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 12:14:10 2021

@author: Raquel
"""

from platypus import *
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
import keras
import tensorflow as tf
import config.config as config

loaded_model = keras.saving.load_model("../models/" + config.MODEL_NAME)


class FS_O1O2O3_LSTM(Problem):

    def __init__(self, test_X, test_y, normCorr, nVar = 2, nobjs = 2):
        super(FS_O1O2O3_LSTM, self).__init__(nVar, nobjs)
        self.types[:] = Binary(1)
        self.test_X = test_X
        self.test_y = test_y
        self.normCorr = normCorr
    
    def evaluate(self, solution):
        mask = np.array([sol[0] for sol in solution.variables], dtype=bool)  
        DprimeTest = np.array(self.test_X)

        # Apply the mask
        DprimeTest[:, ~mask] = 0.0
        N = mask.sum()  # Number of selected attributes
               
        # Correlation
        corr = tuple(np.array(tuple(self.normCorr))[mask])
        if N == 0:
            corr = 0
        else:
            corr = sum(list(corr[:N]))
        
        # Calculate RMSE        
        DprimeTest = DprimeTest.reshape(self.test_X.shape[0], 1, self.test_X.shape[1])
        predTest = loaded_model.predict(DprimeTest)

        # Calculate RMSE
        rmseTest = root_mean_squared_error(self.test_y, predTest.ravel())
        
        solution.objectives[:] = [rmseTest, N, -1*corr]
        