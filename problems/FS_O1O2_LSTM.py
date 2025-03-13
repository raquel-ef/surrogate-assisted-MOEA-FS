# -*- coding: utf-8 -*-
"""
Created on Sat May 21 11:14:57 2022

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


class FS_O1O2_LSTM(Problem):
    def __init__(self, test_X, test_y, nVar=2, nobjs=2):
        super(FS_O1O2_LSTM, self).__init__(nVar, nobjs)
        self.types[:] = Binary(1)
        self.test_X = test_X
        self.test_y = test_y

    def evaluate(self, solution):
        mask = np.array([sol[0] for sol in solution.variables], dtype=bool)  
        DprimeTest = np.array(self.test_X)

        # Apply the mask
        DprimeTest[:, ~mask] = 0.0
        
        # Reshape for model input and predict
        DprimeTest = DprimeTest.reshape(self.test_X.shape[0], 1, self.test_X.shape[1])
        predTest = loaded_model.predict(DprimeTest)

        # Calculate RMSE
        rmseTest = root_mean_squared_error(self.test_y, predTest.ravel())
        N = mask.sum()  # Number of selected attributes

        solution.objectives[:] = [rmseTest, N]
