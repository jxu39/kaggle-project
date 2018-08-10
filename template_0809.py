# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 19:37:20 2018

@author: jiian
"""

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
from sklearn import preprocessing

#-----------------------------------------------------------------------------------------------
#                            self-defined functions
#-----------------------------------------------------------------------------------------------

#----------------------------------- 
#   missing value related functions
#-----------------------------------

# show the percentage of missing values for each column
def missing_values(df, cols):
    for col in cols:
        num_missing_values = df[col].isnull().sum()
        num_rows = len(df)
        percentage_missing_values = 100 * num_missing_values / float(num_rows)
        #print('Percentage of missing values in column {0} = {1} / {2} = {3}%'.format(col, num_missing_values, num_rows, percentage_missing_values))

        # Fill in missing data with its median
def fill_with_median(data_frame):
    """This function will fill the missing value in the data frame with median value of each column"""
    return data_frame.fillna(value=data_frame.median(axis=0, skipna=True))

def fill_with_0(data_frame):
    return data_frame.fillna(0.0)

#----------------------------------- 
#   visualization data
#-----------------------------------
# plot histograms
def plot_histograms(df, cols):
    for col in cols:
        df.hist(column = col, bins = 100) 
 
#----------------------------------- 
#   Normalization
#-----------------------------------

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

# scale
def scale(dataframe):
    dataframe_scale = preprocessing.scale(dataframe)
    return dataframe_scale

#pca
def run_pca(app_train):
    app_train_only_numerics = app_train.select_dtypes(include=[np.number])      
    app_train_only_numerics_drop_target = app_train_only_numerics.drop('TARGET', 1)
    app_train_only_numerics_drop_target_fillna = app_train_only_numerics_drop_target.fillna(0.0)

    pca = PCA(n_components=10)
    pca.fit_transform(app_train_only_numerics_drop_target_fillna)
    print(pca.explained_variance_ratio_)
    mean = pca.components_.mean(axis=0) 
    std = pca.components_.std(axis=0) 
    col = list( app_train_only_numerics_drop_target_fillna)
    mc = np.argpartition(mean, -4)[-4:]
    print ("the most important features:")
    for i in mc:
       print(col[i])
#---------------------------------------------------------------
#    Train model, make prediction
#--------------------------------------------------------------

def do_machine_learning_split(x_train, x_test, y_train, y_test, classifier):
    print('**********************************')
    print('classifier = {0}:'.format(classifier))

       
    # train model
    model = None
    if classifier == 'logistic_regression':
        model = LogisticRegression()
    elif classifier == 'random_forest':
        model = RandomForestClassifier(n_estimators = 10)   
    model.fit(x_train, y_train)

    # prediction
    prediction_proba = model.predict_proba(x_test)[:, 1]

    print('prediction_proba:')
    print(prediction_proba)

    # AUC (Area Under ROC) <- the larger the better.
    auc = roc_auc_score(y_test, prediction_proba)
    print('auc = {0}'.format(auc))
    
   
def do_machine_learning_submit(x_train, y_train, x_test, classifier, ID):
    print('**********************************')
    print('classifier = {0}:'.format(classifier))
       
    # train model
    model = None
    if classifier == 'logistic_regression':
        model = LogisticRegression()
    elif classifier == 'random_forest':
        model = RandomForestClassifier(n_estimators = 10)   
    model.fit(x_train, y_train)

    # prediction
    prediction_proba = model.predict_proba(x_test)[:, 1]

    print('prediction_proba:')
    print(prediction_proba)

    # Submission dataframe
    submit = ID
    submit['TARGET'] = prediction_proba

    # convert to csv
    submit.to_csv('log_reg_' + classifier + '.csv', index = False)
    
def execute(app_train, app_test, feature):
    #-----------------------------------------------------------------------------------------------
    #                            features choosing
    #-----------------------------------------------------------------------------------------------

    #choose features 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'

    app_train_chosen = app_train[feature]
    app_test_chosen = app_test[feature]
    
    #-----------------------------------------------------------------------------------------------
    #                            prepocessing (missing values)
    #-----------------------------------------------------------------------------------------------
    
    # missing values
    # call function to show the percentages of missing values
    missing_values(app_train_chosen, feature)
    missing_values(app_test_chosen, feature)
    app_train_chosen_missing_filled = fill_with_median(app_train_chosen)
    app_test_filled_missing_filled = fill_with_median(app_test_chosen)
    
    # normalization
    app_train_chosen_missing_filled_scaled = scale(app_train_chosen_missing_filled)
    app_test_filled_missing_filled_scaled = scale(app_test_filled_missing_filled)
    
    
    #-------------------------------------------------------------------------------------------------------
    #  Train model (logistic regression) and predict with splitted data
    #-------------------------------------------------------------------------------------------------------
    
    # Split original train data into: train data (80%), test data (20%)
    x_train, x_test = train_test_split(app_train_chosen_missing_filled_scaled, test_size = 0.2)
    y_train, y_test = train_test_split(app_train['TARGET'], test_size = 0.2)
    
    # print out
    original_train_size = len(app_train_chosen_missing_filled_scaled)
    print('original_train_size = {0}'.format(original_train_size)) # 307,511
    print('split_train size = {0}, split_test size = {1}'.format(len(x_train), len(x_test)))
    print('split_train ratio = {0}, split_test ratio = {1}'.format(len(x_train) / original_train_size, len(x_test) / original_train_size))
    
    classifiers = ['logistic_regression', 'random_forest']
    
    for classifier in classifiers:
        do_machine_learning_split(x_train, x_test, y_train, y_test, classifier)
    #-----------------------------------------------------------------------------------------------
    #   Train model (logistic regression) and predict with given test data, T
    #-----------------------------------------------------------------------------------------------
    
    x_train = app_train_chosen_missing_filled_scaled
    x_test = app_test_filled_missing_filled_scaled
    y_train = app_train['TARGET']
    
    ID =  app_test[['SK_ID_CURR']]
    for classifier in classifiers:
        do_machine_learning_submit(x_train, y_train, x_test, classifier, ID)


#-----------------------------------------------------------------------------------------------
#                            load csv data
#-----------------------------------------------------------------------------------------------
    
#Create a pandas dataframe from the csv file. 
app_train = pd.read_csv('application_train.csv')
app_test = pd.read_csv('application_test.csv')

# run pca
run_pca(app_train)

# execute
feature = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
execute(app_train, app_test,feature)


    

