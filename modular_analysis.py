#-----------------------------------------------------------------------------------------------
#                            #Libraries needed to run the tool
#-----------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


# Normlaize
def normalize(df, cols):
    res = {}  # Empty dictionary

    for col in cols:
        df_col = df[col]
        max = df_col.max()
        df_col_norm = df_col / max
        res[col] = df_col_norm

    res_df = pd.DataFrame(data=res)  # Converts dictionary to DataFrame
    return res_df


# Fill in missing data with its median
def fill_with_median(data_frame):
    """This function will fill the missing value in the data frame with median value of each column"""
    return data_frame.fillna(value=data_frame.median(axis=0, skipna=True))


def custom_data_machine_learning(x_train, y_train, x_test, y_test, classifier):
    print('**********************************')
    print('classifier = {0}:'.format(classifier))

    # call normalization function
    x_train_norm = normalize(x_train, x_train.columns.values)
    x_test_norm = normalize(x_test, x_test.columns.values)

    # train model
    model = None

    if classifier == 'logistic_regression':
        model = LogisticRegression()
    elif classifier == 'random_forest':
        model = RandomForestClassifier(n_estimators=10)

    model.fit(x_train_norm, y_train)

    # prediction
    # prediction = model.predict(x_test_norm)
    prediction_proba = model.predict_proba(x_test_norm)[:, 1]

    print('prediction_proba:')
    print(prediction_proba)

    # AUC (Area Under ROC) <- the larger the better.
    # auc = roc_auc_score(y_test, prediction_proba)
    # print('auc = {0}'.format(auc))


def fill_ext():
    num_only = app_train.select_dtypes(include=[np.number])
    num_only_train = num_only.drop(['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'], axis=1)
    filled_num_only_train = fill_with_median(num_only_train)
    ext_src_1 = num_only['EXT_SOURCE_1']
    combined = pd.concat([filled_num_only_train, ext_src_1], axis=1, sort=False)
    cat_notnull_with_ext1 = combined[combined['EXT_SOURCE_1'].notnull()]
    cat_null_with_ext1 = combined[combined['EXT_SOURCE_1'].isnull()]
    x_train = cat_notnull_with_ext1.drop(['EXT_SOURCE_1'], axis=1)
    y_train = cat_notnull_with_ext1['EXT_SOURCE_1']

    x_test = cat_null_with_ext1.drop(['EXT_SOURCE_1'], axis=1)
    y_test = cat_null_with_ext1['EXT_SOURCE_1']

    custom_data_machine_learning(x_train, y_train, x_test, y_test, 'logistic_regression')





# -----------------------------------------------------------------------------------------------
#                            load csv data
# -----------------------------------------------------------------------------------------------

# Create a pandas dataframe from the csv file.
app_train = pd.read_csv('application_train.csv')
app_test = pd.read_csv('application_test.csv')
something()
