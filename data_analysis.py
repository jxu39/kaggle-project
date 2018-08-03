# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#-----------------------------------------------------------------------------------------------
#                            #Libraries needed to run the tool
#-----------------------------------------------------------------------------------------------

import numpy as np 
import pandas as pd 
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
#matplotlib inline

#import plotly
#import plotly.plotly as py
#import plotly.graph_objs as go
#plotly.offline.init_notebook_mode()

from sklearn.linear_model import LogisticRegression
from sklearn import metrics


#-----------------------------------------------------------------------------------------------
#                            self-defined functions
#-----------------------------------------------------------------------------------------------

# plot histograms
def plot_histograms(df, cols):
    for col in cols:
        df.hist(column = col, bins = 100) 
 
# show the percentage of missing values for each column
def missing_values(df, cols):
    for col in cols:
        num_missing_values = df[col].isnull().sum()
        num_rows = len(df)
        percentage_missing_values = 100 * num_missing_values / float(num_rows)
        print('Percentage of missing values in column {0} = {1} / {2} = {3}%'.
              format(col, num_missing_values, num_rows, percentage_missing_values))
# Normlaize
def normalize(df, cols):
    res = {} #Empty dictionary
    
    for col in cols:
        df_col = df[col]
        max = df_col.max()
        df_col_norm = df_col / max
        res[col] = df_col_norm
    
    res_df = pd.DataFrame(data = res) # Converts dictionary to DataFrame    
    return res_df


# Fill in missing data with its median
def fill_with_median(data_frame):
    """This function will fill the missing value in the data frame with median value of each column"""
    return data_frame.fillna(value=data_frame.median(axis=0, skipna=True))


#-----------------------------------------------------------------------------------------------
#                            load csv data
#-----------------------------------------------------------------------------------------------
    
#Create a pandas dataframe from the csv file. 
app_train = pd.read_csv('application_train.csv')
app_test = pd.read_csv('application_test.csv')

#Print some rows
#app_train.head(3)
#app_test.head(3)

#-----------------------------------------------------------------------------------------------
#                            features choosing
#-----------------------------------------------------------------------------------------------

#choose features 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
app_train_ext = app_train[['SK_ID_CURR', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'TARGET']]
app_test_ext = app_test[['SK_ID_CURR', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
ext_source_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

#-----------------------------------------------------------------------------------------------
#                            prepocessing (missing values)
#-----------------------------------------------------------------------------------------------

# call function to show the percentages of missing values
missing_values(app_train_ext, ext_source_cols)
missing_values(app_test_ext, ext_source_cols)

# option 1: filling missing values with 0
# app_train_ext_fillna = app_train_ext.fillna(0.0)
# app_test_ext_fillna = app_test_ext.fillna(0.0)

# option 2: remove missing values
# app_train_ext_removed_nan = app_train_ext.dropna(subset = ext_source_cols)
# app_test_ext_removed_nan = app_test_ext.dropna(subset = ext_source_cols)

# option 3: fill missing values with the median values
ext_train_filled_median = fill_with_median(app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']])
ext_test_filled_median = fill_with_median(app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']])

#-----------------------------------------------------------------------------------------------
#                            Train model (logistic regression) and predict
#-----------------------------------------------------------------------------------------------

# preparation
x_train = ext_train_filled_median
y_train = app_train[['TARGET']]
x_test = ext_test_filled_median

# call normalization function
x_train_norm = normalize(x_train, ext_source_cols)
x_test_norm = normalize(x_test, ext_source_cols)

# train model
logreg = LogisticRegression()
logreg.fit(x_train_norm, y_train)

# prediction
y_pred = logreg.predict(x_test_norm)
logreg_prediction = logreg.predict_proba(x_test_norm)[:, 1]

# Submission dataframe
submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = logreg_prediction
#submit.head()

# convert to csv
submit.to_csv('log_reg.csv', index = False)
