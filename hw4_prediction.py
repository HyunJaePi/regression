#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:44:19 2020

hw4 code 2 -- 
Using L1 regularization and  Isolation Forest, 
this program predicts output values from test datasets.

@author: hyunjaepi, hyunpi@brandeis.edu
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
from numpy import savetxt

import warnings
warnings.filterwarnings('ignore')

# identify inlier indices using Isolation Forest (outlier detection)
def indices_inliers_by_isolation_forest(data, contamination_factor):
    iso = IsolationForest(contamination=contamination_factor)
    indices_outliers = iso.fit_predict(data)
    mask = indices_outliers != -1 # select all rows that are not outliers
    return mask

# read train and test data 
def read_train_and_test_files():    
    folder1 = ['a','b']
    folder2 = ['1','2','3','4','5']
    str0 = './HW4data/'

    filename_train = []
    filename_test = []
    for f1 in folder1:
        if f1 == 'a':
            str_tmp =""
        else:
            str_tmp ="X"
            
        for f2 in folder2:
                filename_train.append(str0 + f1 + '/' + f2 + '/' + 'Train'+str_tmp+'_'+f2+f1+'.csv')
                filename_test.append(str0 + f1 + '/' + f2 + '/' + 'Test'+'_'+f2+f1+'.csv')
            
    return [filename_train, filename_test]

# predict outputs from test datasets
def prediction_on_test_data(filenames, const_regularization, const_outlier_detection):
    # file names
    filename_train = filenames[0];
    filename_test = filenames[1];
    
    # read training data
    data = pd.read_csv(filename_train).to_numpy()
    X = data[:, 0:-1]
    y = data[:, -1]
    
    # outlier detection
    mask = indices_inliers_by_isolation_forest(X, const_outlier_detection)
    x_train, y_train = X[mask, :], y[mask]
    
    # L1 regression w/ regularization
    lasso = Lasso(alpha = const_regularization)
    lasso.fit(x_train, y_train)
 
    # prediction on test dataset
    x_test = pd.read_csv(filename_test).to_numpy()        
    y_pred = lasso.predict(x_test)
    
    return y_pred


# run & save results
const_regularization = [.02, .03, .01, .01, .01, .01, .02, .01, .03, .01]
const_outlier_detection = [.25, .03, .47, .44, .37, .48, .47, .00, .05, .28]
filenames = read_train_and_test_files()

predictions = np.empty([15, 10])
i = 0;
for i in range(10):
    predictions[:, i] = prediction_on_test_data([filenames[0][i],filenames[1][i]], const_regularization[i], const_outlier_detection[i])
    i = i + 1

savetxt('results.csv', predictions, delimiter=',')   

