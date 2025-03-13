# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 12:14:10 2021

@author: Raquel
"""

from platypus import Problem, Binary
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor


class RF_wrapper(Problem):

    def __init__(self, train_X, train_y, test_X, test_y, nVar = 2, nobjs = 2):
        super(RF_wrapper, self).__init__(nVar, nobjs)
        self.types[:] = Binary(1)
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y


    def evaluate(self, solution):
        # Convert solution variables to a mask (binary selection of features)
        mask = np.array([sol[0] for sol in solution.variables], dtype=bool)
        
        # Select features based on the mask
        trainX_sel = np.array(self.train_X)[:, mask]
        testX_sel = np.array(self.test_X)[:, mask]
        
        N = trainX_sel.shape[1]  # Number of selected features

        # If no features are selected, use mean prediction (no training)
        if N == 0:
            predTest = np.mean(self.train_y) * np.ones_like(self.test_y)
            rmseTest = root_mean_squared_error(self.test_y, predTest)
        else:
            # Train the model
            model = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=0).fit(trainX_sel, self.train_y)
            predTest = model.predict(testX_sel)
            rmseTest = root_mean_squared_error(self.test_y, predTest)
        
        # Assign objectives (RMSE and number of features)
        solution.objectives[:] = [rmseTest, N]

