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

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

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
        #print('Percentage of missing values in column {0} = {1} / {2} = {3}%'.format(col, num_missing_values, num_rows, percentage_missing_values))

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
ext_train_filled_median = fill_with_median(app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'TARGET']])
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

#-----------------------------------------------------------------------------------------------
#                            Split original train data into train and test data
#-----------------------------------------------------------------------------------------------

original_train_size = len(ext_train_filled_median)
print('original_train_size = {0}'.format(original_train_size)) # 307,511

# Split original train data into: train data (80%), test data (20%)
split_train, split_test = train_test_split(ext_train_filled_median, test_size = 0.2)
print('split_train size = {0}, split_test size = {1}'.format(len(split_train), len(split_test)))
print('split_train ratio = {0}, split_test ratio = {1}'.format(len(split_train) / original_train_size, len(split_test) / original_train_size))


#-----------------------------------------------------------------------------------------------
#                            Train model, make prediction and output test metric
#-----------------------------------------------------------------------------------------------

def do_machine_learning(train_data, test_data, classifier):
    print('**********************************')
    print('classifier = {0}:'.format(classifier))

    x_train = train_data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
    y_train = train_data[['TARGET']]
    x_test = test_data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
    y_test = test_data[['TARGET']]

    # call normalization function
    x_train_norm = normalize(x_train, ext_source_cols)
    x_test_norm = normalize(x_test, ext_source_cols)

    # train model
    model = None

    if classifier == 'logistic_regression':
        model = LogisticRegression()
    elif classifier == 'random_forest':
        model = RandomForestClassifier(n_estimators = 10)
        
    model.fit(x_train_norm, y_train)

    # prediction
    # prediction = model.predict(x_test_norm)
    prediction_proba = model.predict_proba(x_test_norm)[:, 1]

    print('prediction_proba:')
    print(prediction_proba)

    # AUC (Area Under ROC) <- the larger the better.
    auc = roc_auc_score(y_test, prediction_proba)
    print('auc = {0}'.format(auc))

classifiers = ['logistic_regression', 'random_forest']

for classifiers in classifiers:
    do_machine_learning(split_train, split_test, classifiers)

