# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 12:14:10 2021

@author: Raquel
"""

from platypus import *
import pandas as pd
import numpy as np


class FS_O3O4O2_LSTM(Problem):

    def __init__(self, normCorr, reliefFscores, nVar = 2, nobjs = 2):
        super(FS_O3O4O2_LSTM, self).__init__(nVar, nobjs)
        self.types[:] = Binary(1)
        self.normCorr = normCorr
        self.reliefFscores = reliefFscores
    
    def evaluate(self, solution):
        mask = np.array([sol[0] for sol in solution.variables], dtype=bool)  
        N = mask.sum()  # Number of selected attributes
         
        # Correlation
        corr = tuple(np.array(tuple(self.normCorr))[mask])
        if N == 0:
            corr = 0
        else:
            corr = sum(list(corr[:N]))
            
        # Apply ReliefF
        rScores = tuple(np.array(tuple(self.reliefFscores))[mask])
        if N == 0:
            rs = 0 # assign the worse value
        else: 
            rs = sum(list(rScores[:N]))
            
        
        solution.objectives[:] = [-1*corr, -1*rs, N] # *-1 to transform MAXIMIZATION objective into MINIMIZATION
        