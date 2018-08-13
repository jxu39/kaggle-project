'''
This code makes the pipeline more object oriented and more general, which can also be used in other projects.

The current output:

Accuracies (AUC):
Logistic Regression: 0.7180776485070874
Neural Network: 0.7199212385013528
KNN: 0.5981104854444489
Decision Tree: 0.5304570916426096
Linear Classifier with SGD: 0.7178525754160756

todo: 
1. Replace some functions (like normalize) by standard libarary functions.
2. Complete the code of filling missing values with prediction. 
3. Deal with categorical columns and include them as features.
4. Gather all features and do PCA on them.
5. Explore other tables.
'''

import numpy as np 
import pandas as pd 
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import preprocessing

# Shared parameters
class Parameters:
    def __init__(self):
        self.train_table_name = 'application_train.csv'
        self.test_table_name = 'application_test.csv'
        self.feature_names = None
        self.id_name = 'SK_ID_CURR'
        self.target_name = None
        self.test_ratio = 0.3
        

# Preprocesses data: normalize, fill in missing values with median
class Preprocessor(Parameters):
    def __init__(self, feature_names, target_name, is_for_filling_nan):
        Parameters.__init__(self)
        self.feature_names = feature_names
        self.target_name = target_name
        self.is_for_filling_nan = is_for_filling_nan
    '''
    def normalize(self, df):
        res = {} #Empty dictionary
        
        for col in df.columns:
            df_col = df[col]
            max = df_col.max()
            df_col_norm = df_col / max
            res[col] = df_col_norm
        
        res_df = pd.DataFrame(data = res) # Converts dictionary to DataFrame    
        return res_df
    '''
    def normalize(self, df)
        return preprocessing.scale(df)
        
    def transform(self, df):
        df_features = df[self.feature_names]

        # Fill nan values with median
        df_features_fillednan = df_features.fillna(df_features.median())

        # Normalize
        df_features_normalized = self.normalize(df_features_fillednan)

        # Add target column back, since test data needs it
        df_target = df[[self.target_name]]
        df_with_target = pd.merge(df_features_normalized, df_target, left_index = True, right_index = True)

        # Split input data into train and test data
        train = None
        test = None

        if self.is_for_filling_nan:
            train = df_with_target[df_with_target[self.target_name].notnull()]
            test = df_with_target[df_with_target[self.target_name].isnull()]
        else:
            train, test = train_test_split(df_with_target, test_size = self.test_ratio)

        return (train, test)

class Learner(Parameters):
    def __init__(self, train, test, models, feature_names, target_name, is_for_filling_nan):
        Parameters.__init__(self)
        self.train = train
        self.test = test
        self.models = models
        self.feature_names = feature_names
        self.target_name = target_name
        self.is_for_filling_nan = is_for_filling_nan
        
    def transform(self):
        for name, model in self.models.items():
            # print(self.is_for_filling_nan, self.feature_names)
            model.fit(self.train[self.feature_names], self.train[self.target_name])
            prediction_proba = None

            # AUC (Area Under ROC) <- the larger the better.
            if self.is_for_filling_nan:
                prediction = model.predict(self.test[self.feature_names])
                print("For filling, prediction:")
                print(prediction)
            else:
                prediction_proba = model.predict_proba(self.test[self.feature_names])[:, 1]
                auc = roc_auc_score(self.test[self.target_name], prediction_proba)
                print("{0}: {1}".format(name, auc))

# Fill in missing values using linear classifier model:
class MissingValueFiller(Parameters):
    def __init__(self, feature_names, target_name):
        Parameters.__init__(self)
        self.feature_names = feature_names
        self.target_name = target_name

    def transform(self, df):
        preprocesser = Preprocesser(self.feature_names, self.target_name, True)
        (train, test) = preprocesser.transform(df)

        # Define models:
        models_dic = {}
        models_dic['Linear Regression'] = linear_model.LinearRegression()

        # Learn and predict
        learner = Learner(train, test, models_dic, self.feature_names, self.target_name, True)
        learner.transform()
        
# use PCA to obtain feature importance:
class PCA(Parameters):
    def __init__(self):
        Parameters.__init__(self)

    def transform(self):
        # choose only numerical columns
        app_train_only_numerics = self.app_train.select_dtypes(include=[np.number])      
        app_train_only_numerics_drop_target = app_train_only_numerics.drop('TARGET', 1)
        app_train_only_numerics_drop_target_fillna = app_train_only_numerics_drop_target.fillna(0.0)
        
        # run pca
        pca = PCA(n_components=10)
        pca.fit_transform(app_train_only_numerics_drop_target_fillna)
        
        # print out the explained variance ratio
        print("The explained variance ratios are: ")
        print(pca.explained_variance_ratio_)
        
        # heuristic approach to sort the features importances
        mean = pca.components_.mean(axis=0) 
        std = pca.components_.std(axis=0) 
        col = list( app_train_only_numerics_drop_target_fillna)
        mc = np.argpartition(mean, -4)[-4:]
        print ("the most important features:")
        for i in mc:
           print(col[i])      

class Application(Parameters):
    def __init__(self):
        Parameters.__init__(self)
        
    def run(self):
        # Load data:
        data = pd.read_csv(self.train_table_name)   

        # Fill in missing values from model prediction
        missing_value_filler = MissingValueFiller(['DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'AMT_CREDIT'], 'EXT_SOURCE_1')
        missing_value_filler.transform(data)

        # Preprocess:     
        feature_names_list = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        target_name_str = 'TARGET'
        preprocesser = Preprocesser(feature_names_list, target_name_str, False)
        (train, test) = preprocesser.transform(data)

        # Define models:
        models_dic = {}
        models_dic['Logistic Regression'] = LogisticRegression()
        models_dic['Neural Network'] = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 10), random_state=1)
        models_dic['KNN'] = KNeighborsClassifier()
        models_dic['Decision Tree'] = DecisionTreeClassifier()
        models_dic['Linear Classifier with SGD'] = linear_model.SGDClassifier(loss = 'modified_huber')

        # Learn and predict
        learner = Learner(train, test, models_dic, feature_names_list, target_name_str, False)
        print("Accuracies (AUC):")
        learner.transform()
        
application = Application()
application.run()
