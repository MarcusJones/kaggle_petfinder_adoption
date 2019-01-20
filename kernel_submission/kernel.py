#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 10:32:09 2018

@author: m.jones
"""

#%%
#!pip install git+https://github.com/MarcusJones/kaggle_utils.git

#%%
import os
# %% Globals
#
# LANDSCAPE_A3 = (16.53, 11.69)
# PORTRAIT_A3 = (11.69, 16.53)
# LANDSCAPE_A4 = (11.69, 8.27)
if 'KAGGLE_WORKING_DIR' in os.environ:
    DEPLOYMENT = 'Kaggle'
else:
    DEPLOYMENT = 'Local'

if DEPLOYMENT=='Kaggle':
    PATH_DATA_ROOT = r"~"
    SAMPLE_FRACTION = 1
if DEPLOYMENT == 'Local':
    PATH_DATA_ROOT = r"~/DATA/petfinder_adoption"
    SAMPLE_FRACTION = 0.5


# PATH_OUT = r"/home/batman/git/hack_sfpd1/Out"
# PATH_OUT_KDE = r"/home/batman/git/hack_sfpd1/out_kde"
# PATH_REPORTING = r"/home/batman/git/hack_sfpd1/Reporting"
# PATH_MODELS = r"/home/batman/git/hack_sfpd4/models"
# TITLE_FONT = {'fontname': 'helvetica'}


# TITLE_FONT_NAME = "Arial"
# plt.rc('font', family='Helvetica')

#%% ===========================================================================
# Standard imports
# =============================================================================
import os
from pathlib import Path
import sys
import zipfile
from datetime import datetime
import gc
import time

#%% ===========================================================================
# ML imports
# =============================================================================
import numpy as np
import pandas as pd
import sklearn as sk

import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.pipeline
import sklearn.model_selection

from sklearn_pandas import DataFrameMapper

# Models
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Metric
from sklearn.metrics import cohen_kappa_score
def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

#%% ===========================================================================
# Custom imports
# =============================================================================
import kaggle_utils.transformers as trf

#%% ===========================================================================
# Logging
# =============================================================================
import logging
#Delete Jupyter notebook root logger handler
logger = logging.getLogger()
logger.handlers = []

# Set level
logger.setLevel(logging.DEBUG)

# Create formatter
#FORMAT = "%(asctime)s - %(levelno)-3s - %(module)-10s  %(funcName)-10s: %(message)s"
#FORMAT = "%(asctime)s - %(levelno)-3s - %(funcName)-10s: %(message)s"
#FORMAT = "%(asctime)s - %(funcName)-10s: %(message)s"
FORMAT = "%(asctime)s : %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
#DATE_FMT = "%H:%M:%S"
formatter = logging.Formatter(FORMAT, DATE_FMT)

# Create handler and assign
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
logger.handlers = [handler]
logging.debug("Logging started")



#%% ===========================================================================
# Data source and paths
# =============================================================================
path_data = Path(PATH_DATA_ROOT, r"").expanduser()
assert path_data.exists()
logging.info("Data path {}".format(PATH_DATA_ROOT))

#%% ===========================================================================
# Load data
# =============================================================================
logging.info(f"Loading files into memory")

# def load_zip
# with zipfile.ZipFile(path_data / "train.zip").open("train.csv") as f:
#     df_train = pd.read_csv(f, delimiter=',')
# with zipfile.ZipFile(path_data / "test.zip").open("test.csv") as f:
#     df_test = pd.read_csv(f, delimiter=',')

df_train = pd.read_csv(path_data / 'train.csv')
df_test = pd.read_csv(path_data / 'test' / 'test.csv')

breeds = pd.read_csv(path_data / "breed_labels.csv")
colors = pd.read_csv(path_data / "color_labels.csv")
states = pd.read_csv(path_data / "state_labels.csv")

logging.debug("Loaded train {}".format(df_train.shape))
logging.debug("Loaded test {}".format(df_test.shape))

# Add a column to label the source of the data
df_train['dataset_type'] = 'train'
df_test['dataset_type'] = 'test'

logging.debug("Added dataset_type column for origin".format())
df_all = pd.concat([df_train, df_test], sort=False)
df_all.set_index('PetID',inplace=True)

del df_train, df_test

#%% Memory of the training DF:
logging.debug("Size of df_all: {} MB".format(sys.getsizeof(df_all) / 1000 / 1000))

#%%
df_all['PhotoAmt'] = df_all['PhotoAmt'].astype('int')

#%% Category Mappings
label_maps = dict()
label_maps['Vaccinated'] = {
    1 : 'Yes',
    2 : 'No',
    3 : 'Not sure',
}
label_maps['Type'] = {
    1:"Dog",
    2:"Cat"
}
label_maps['AdoptionSpeed'] = {
    0 : "same day",
    1 : "between 1 and 7 days",
    2 : "between 8 and 30 days",
    3 : "between 31 and 90 days",
    4 : "No adoption after 100 days",
}
label_maps['Gender'] = {
    1 : 'Male',
    2 : 'Female',
    3 : 'Group',
}
label_maps['MaturitySize'] = {
    1 : 'Small',
    2 : 'Medium',
    3 : 'Large',
    4 : 'Extra Large',
    0 : 'Not Specified',
}
label_maps['FurLength'] = {
    1 : 'Short',
    2 : 'Medium',
    3 : 'Long',
    0 : 'Not Specified',
}
label_maps['Dewormed'] = {
    1 : 'Yes',
    2 : 'No',
    3 : 'Not sure',
}
label_maps['Sterilized'] = {
    1 : 'Yes',
    2 : 'No',
    3 : 'Not sure',
}
label_maps['Health'] = {
    1 : 'Healthy',
    2 : 'Minor Injury',
    3 : 'Serious Injury',
    0 : 'Not Specified',
}

# For the breeds, load the two types seperate
dog_breed = breeds[['BreedID','BreedName']][breeds['Type']==1].copy()
map_dog_breed = dict(zip(dog_breed['BreedID'], dog_breed['BreedName']))

cat_breed = breeds[['BreedID','BreedName']][breeds['Type']==2].copy()
map_cat_breed = dict(zip(cat_breed['BreedID'], cat_breed['BreedName']))

# Just in case, check for overlap in breeds
# for i in range(308):
#     print(i,end=": ")
#     if i in map_dog_breed: print(map_dog_breed[i], end=' - ')
#     if i in map_cat_breed: print(map_cat_breed[i], end=' - ')
#     if i in map_dog_breed and i in map_cat_breed: raise
#     print()

# It's fine, join them into one dict
map_all_breeds = dict()
map_all_breeds.update(map_dog_breed)
map_all_breeds.update(map_cat_breed)
map_all_breeds[0] = "NA"

# Now add them to the master label dictionary for each column
label_maps['Breed1'] = map_all_breeds
label_maps['Breed2'] = map_all_breeds

# Similarly, load the color map
map_colors = dict(zip(colors['ColorID'], colors['ColorName']))
map_colors[0] = "NA"
label_maps['Color1'] = map_colors
label_maps['Color2'] = map_colors
label_maps['Color3'] = map_colors

# And the states map
label_maps['State'] = dict(zip(states['StateID'], states['StateName']))

logging.debug("Category mappings for {} columns created".format(len(label_maps)))

for map in label_maps:
    print(map, label_maps[map])

#%% Dynamically create the transformation definitions
tx_definitions = [(col_name, trf.NumericalToCat(label_maps)) for col_name in label_maps]

#%% Pipeline
# Build the pipeline
# NOTES:
# input_df - Ensure the passed in column enters as a series or DF
# df_out - Ensure the pipeline returns a df
# default - if a column is not transformed, keep it unchanged!
# WARNINGS:
# The categorical dtype is LOST!
# Do NOT use DataFrameMapper for creating new columns, use a regular pipeline!
data_mapper = DataFrameMapper(
    tx_definitions,
input_df=True, df_out=True, default=None)

print("DataFrameMapper, applies transforms directly selected columns")
for i, step in enumerate(data_mapper.features):
    print(i, step)

#%% FIT TRANSFORM
df_all = data_mapper.fit_transform(df_all)

logging.debug("Size of train df_all with string columns: {} MB".format(sys.getsizeof(df_all)/1000/1000))
#%% WARNING - sklearn-pandas has a flaw, it does not preserve categorical features!!!
for col in label_maps:
    print(col)
    df_all[col] = df_all[col].astype('category')
logging.debug("Reapplied categorical features".format())
logging.debug("Size of df_all with categorical features: {} MB".format(sys.getsizeof(df_all)/1000/1000))


#%% SUMMARY

logging.debug("Final df_all {}".format(df_all.shape))
#%% DONE HERE - DELETE UNUSED
print("******************************")

del_vars =[
    'breeds',
    'cat_breed',
    'colors',
    'data_mapper',
    'dog_breed',
    'map_colors',
    'map_all_breeds',
    'map_cat_breed',
    'map_dog_breed',
    'states',
]
cnt = 0
for name in dir():
    if name in del_vars:
        cnt+=1
        del globals()[name]
logging.info(f"Removed {cnt} variables from memory")
del cnt, name, del_vars
# %% ===========================================================================
# Feature
# =============================================================================
def pure_breed(row):
    # print(row)
    mixed_breed_keywords = ['domestic', 'tabby', 'mixed']

    # Mixed if labelled as such
    if row['Breed1'] == 'Mixed Breed':
        return False

    # Possible pure if no second breed
    elif row['Breed2'] == 'NA':
        # Reject domestic keywords
        if any([word in row['Breed1'].lower() for word in mixed_breed_keywords]):
            return False
        else:
            return True
    else:
        return False

#%% Build the pipeline
this_pipeline = sk.pipeline.Pipeline([
        ('counr', trf.MultipleToNewFeature(['Breed1','Breed2'], 'Pure Breed', pure_breed)),
        ])

logging.info("Created pipeline:")
for i, step in enumerate(this_pipeline.steps):
    print(i, step[0], step[1].__str__())

#%% Fit Transform
original_cols = df_all.columns
df_all = this_pipeline.fit_transform(df_all)
logging.debug("Pipeline complete. {} new columns.".format(len(df_all.columns) - len(original_cols)))


#%%

# # sample = df_all.iloc[0:10][['Breed1','Breed2']]
# df_all['Pure Breed'] = df_all.apply(pure_breed,axis=1)
# df_all['Pure Breed'] = df_all['Pure Breed'].astype('category')
# df_all.columns
# df_all.info()
# # For inspection:
# # df_breeds = df_all[['Breed1','Breed2','Pure Breed']]

#%%



#%%
# r = df_all.sample(10)[['Type']]
# len(r)
# r[:] = 1

#
#
#
#
# this_pipeline = sk.pipeline.Pipeline([
#         ('counr', WordCounter('Breed2', 'newcol')),
#         ])
#
# # data_mapper2 = DataFrameMapper(
# #     (['Breed1', 'Breed2'], NumericalToCat(None)),
# #     input_df=True, df_out=True, default=None)
#
# logging.info("Created pipeline:")
# for i, step in enumerate(this_pipeline.steps):
#     print(i, step[0], step[1].__str__()[:60])
#
# #%%
# # transformer_def_list = [
# #     (['Breed1', 'Breed2'], MultipleToNewFeature('Test', pure_breed)),
# #     # (['Breed1', 'Breed2'], PureBreed()),
# #
# # ]
# #
# # transformer_def_list = [
# #     (['Breed2'], WordCounter('Breed2', 'newcol')),
# #     # (['Breed1', 'Breed2'], PureBreed()),
# #
# # ]
# #
# # data_mapper2 = DataFrameMapper(transformer_def_list, input_df=True, df_out=True, default=None)
# df_s = df_all.sample(10)[['Breed1', 'Breed2', 'Type']]
#
# this_pipeline = sk.pipeline.Pipeline([
#         ('counr', WordCounter('Breed2', 'newcol')),
#         ])
#
# # data_mapper2 = DataFrameMapper(
# #     (['Breed1', 'Breed2'], NumericalToCat(None)),
# #     input_df=True, df_out=True, default=None)
#
# logging.info("Created pipeline:")
# for i, step in enumerate(this_pipeline.steps):
#     print(i, step[0], step[1].__str__()[:60])
#
# #%% FIT TRANSFORM
# df_s2 = this_pipeline.fit_transform(df_s)
#
# #%%
#
#
#%%
# The final selection of columns from the main DF
cols_to_use = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength',
               'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'RescuerID', 'VideoAmt',
               'PhotoAmt', 'AdoptionSpeed', 'No_name', 'Pure_breed', 'health', 'Free',
               'score', 'magnitude']

cols_to_discard = [
    'RescuerID',
    'Description',
    'Name',
]


logging.debug("Feature selection".format())
original_columns = df_all.columns
# col_selection = [col for col in all_columns if col not in cols_to_discard]

df_all.drop(cols_to_discard,inplace=True, axis=1)

logging.debug("Selected {} of {} columns".format(len(df_all.columns),len(original_columns)))
logging.debug("Size of df_all with selected features: {} MB".format(sys.getsizeof(df_all)/1000/1000))

logging.debug("Record selection (sampling)".format())
logging.debug("Sampling fraction: {}".format(SAMPLE_FRACTION))
df_all = df_all.sample(frac=SAMPLE_FRACTION)
logging.debug("Final size of data frame: {}".format(df_all.shape))
logging.debug("Size of df_all with selected features and records: {} MB".format(sys.getsizeof(df_all)/1000/1000))

#%%
df_tr = df_all[df_all['dataset_type']=='train'].copy()
df_tr.drop('dataset_type', axis=1, inplace=True)
df_te = df_all[df_all['dataset_type']=='test'].copy()
df_te.drop('dataset_type', axis=1, inplace=True)

y_tr = df_tr['AdoptionSpeed']
logging.debug("y_tr {}".format(y_tr.shape))

X_tr = df_tr.drop(['AdoptionSpeed'], axis=1)
logging.debug("X_tr {}".format(X_tr.shape))

X_te = df_te.drop(['AdoptionSpeed'], axis=1)
logging.debug("X_te {}".format(X_te.shape))

#%% DONE HERE - DELETE UNUSED
print("******************************")

del_vars =[
    'df_all',
    'df_tr',
    'df_te',
]
cnt = 0
for name in dir():
    if name in del_vars:
        cnt+=1
        del globals()[name]
logging.info(f"Removed {cnt} variables from memory")
del cnt, name, del_vars

# Train 2 seperate models, one for cats, one for dogs!!
#%%
n_fold = 5
folds = sk.model_selection.StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=15)

#%%

def train_model(X, X_test, y, params, folds, model_type, plot_feature_importance=False,
                averaging='usual', make_oof=False):

    logging.debug("Starting training {}".format(model_type))
    result_dict = {}
    if make_oof:
        oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        gc.collect()
        print('Fold', fold_n + 1, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            # logging.debug("Categorical column selection here!! TODO: NB".format())
            # cat_cols = X.columns.to_list
            # X_tr.columns.to_list()

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid)

            model = lgb.train(params,
                              train_data,
                              num_boost_round=2000,
                              valid_sets=[train_data, valid_data],
                              verbose_eval=100,
                              early_stopping_rounds=200)

            del train_data, valid_data

            y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration).argmax(1)
            del X_valid
            gc.collect()
            y_pred = model.predict(X_test, num_iteration=model.best_iteration).argmax(1)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                              verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test), ntree_limit=model.best_ntree_limit)

        if model_type == 'lcv':
            model = LogisticRegressionCV(scoring='neg_log_loss', cv=3, multi_class='multinomial')
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        if model_type == 'cat':
            model = CatBoostClassifier(iterations=20000, loss_function='MultiClass', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True,
                      verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test).reshape(-1, )

        if make_oof:
            oof[valid_index] = y_pred_valid.reshape(-1, )

        scores.append(kappa(y_valid, y_pred_valid))
        print('Fold kappa:', kappa(y_valid, y_pred_valid))
        print('')

        if averaging == 'usual':
            prediction += y_pred
        elif averaging == 'rank':
            prediction += pd.Series(y_pred).rank().values

        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importance()
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    if model_type == 'lgb':

        if plot_feature_importance:
            feature_importance["importance"] /= n_fold
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            # plt.figure(figsize=(16, 12));
            # sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            # plt.title('LGB Features (avg over folds)');

            result_dict['feature_importance'] = feature_importance

    result_dict['prediction'] = prediction
    if make_oof:
        result_dict['oof'] = oof

    return result_dict
#

#%%
params = {'num_leaves': 128,
        #  'min_data_in_leaf': 60,
         'objective': 'multiclass',
         'max_depth': -1,
         'learning_rate': 0.05,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 3,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
        #  "lambda_l1": 0.1,
         # "lambda_l2": 0.1,
         "random_state": 42,
         "verbosity": -1,
         "num_class": 5}

#%%

# X_tr.info()
# y_tr.astype('int')
# y_tr.dtype
# y_factors = y_tr.factorize()[0]
y_integers = y_tr.cat.codes
result_dict_lgb = train_model(X=X_tr,
                              X_test=X_te,
                              y=y_integers,
                              params=params,
                              folds=folds,
                              model_type='lgb',
                              plot_feature_importance=True,
                              make_oof=True)


#%% RESULTS
# r = result_dict_lgb['feature_importance']

# cols = result_dict_lgb['feature_importance'][["feature", "importance"]].groupby("feature").mean().sort_values(
#                 by="importance", ascending=False)[:50].index
#
# best_features = result_dict_lgb['feature_importance'].loc[result_dict_lgb['feature_importance'].feature.isin(cols)]
#
# p = plt.figure(figsize=(16, 12))
# sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
# plt.title('LGB Features (avg over folds)')
# plt.show()#%% Open the submission
# with zipfile.ZipFile(path_data / "test.zip").open("sample_submission.csv") as f:
#     df_submission = pd.read_csv(f, delimiter=',')
df_submission = pd.read_csv(path_data / 'test' / 'sample_submission.csv', delimiter=',')


#%% Collect predicitons
prediction = (result_dict_lgb['prediction'])
submission = pd.DataFrame({'PetID': df_submission.PetID, 'AdoptionSpeed': [int(i) for i in prediction]})
submission.head()

#%% Create csv
submission.to_csv('submission.csv', index=False)