#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 10:32:09 2018

@author: m.jones
"""

#%%
#!pip install git+https://github.com/MarcusJones/kaggle_utils.git

#%% ===========================================================================
# Logging
# =============================================================================
import sys
import logging

#Delete Jupyter notebook root logger handler
logger = logging.getLogger()
logger.handlers = []

# Set level
logger.setLevel(logging.INFO)

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
logging.info("Logging started")



#%%
import os
from pathlib import Path
import importlib.util
# %% Globals
#
# LANDSCAPE_A3 = (16.53, 11.69)
# PORTRAIT_A3 = (11.69, 16.53)
# LANDSCAPE_A4 = (11.69, 8.27)
if 'KAGGLE_WORKING_DIR' in os.environ:
    DEPLOYMENT = 'Kaggle'
else:
    DEPLOYMENT = 'Local'
logging.info("Deployment: {}".format(DEPLOYMENT))
if DEPLOYMENT=='Kaggle':
    PATH_DATA_ROOT = Path.cwd() / '..' / 'input'
    SAMPLE_FRACTION = 1
    import transformers as trf
if DEPLOYMENT == 'Local':
    PATH_DATA_ROOT = r"~/DATA/petfinder_adoption"
    PATH_KAGGLE_UTILS = Path(r"../../../kaggle_utils/kaggle_utils").absolute().resolve()
    logging.info("PATH_KAGGLE_UTILS={}".format(PATH_KAGGLE_UTILS))
    sys.path.append(PATH_KAGGLE_UTILS)
    import kaggle_utils.transformers as trf
    SAMPLE_FRACTION = 1


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
from pprint import pprint

#%% ===========================================================================
# ML imports
# =============================================================================
import numpy as np
print('numpy', np.__version__)
import pandas as pd
print('pandas', pd.__version__)
import sklearn as sk
print('sklearn', sk.__version__)

import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.pipeline
import sklearn.model_selection

from sklearn_pandas import DataFrameMapper

# Models
import lightgbm as lgb
print("lightgbm", lgb.__version__)
import xgboost as xgb
print("xgboost", xgb.__version__)
# from catboost import CatBoostClassifier
import catboost as catb
print("catboost", catb.__version__)

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



#%% ===========================================================================
# Data source and paths
# =============================================================================
path_data = Path(PATH_DATA_ROOT, r"").expanduser()
assert path_data.exists(), "Data path does not exist: {}".format(path_data)
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

df_train = pd.read_csv(path_data / 'train'/ 'train.csv')
df_train.set_index(['PetID'],inplace=True)
df_test = pd.read_csv(path_data / 'test' / 'test.csv')
df_test.set_index(['PetID'],inplace=True)

breeds = pd.read_csv(path_data / "breed_labels.csv")
colors = pd.read_csv(path_data / "color_labels.csv")
states = pd.read_csv(path_data / "state_labels.csv")

logging.info("Loaded train {}".format(df_train.shape))
logging.info("Loaded test {}".format(df_test.shape))

# Add a column to label the source of the data
df_train['dataset_type'] = 'train'
df_test['dataset_type'] = 'test'

# Set this aside for debugging
#TODO: Remove later
original_y_train = df_train['AdoptionSpeed'].copy()

logging.info("Added dataset_type column for origin".format())
df_all = pd.concat([df_train, df_test], sort=False)
# df_all.set_index('PetID',inplace=True)

del df_train, df_test

#%% Memory of the training DF:
logging.info("Size of df_all: {} MB".format(sys.getsizeof(df_all) / 1000 / 1000))

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

logging.info("Category mappings for {} columns created".format(len(label_maps)))

for map in label_maps:
    print(map, label_maps[map])



#%% Dynamically create the transformation definitions
# tx_definitions_preview = [(col_name, label_maps[col_name]) for col_name in label_maps]
# for t in tx_definitions_preview:
#     print(t)
tx_definitions = [(col_name, trf.NumericalToCat(label_maps[col_name], True)) for col_name in label_maps]
# col_name = 'Vaccinated'
#%% Pipeline
# Build the pipeline
# NOTES:
# input_df - Ensure the passed in column enters as a series or DF
# df_out - Ensure the pipeline returns a df
# default - if a column is not transformed, keep it unchanged!
# WARNINGS:
# The categorical dtype is LOST!
# The mapping does NOT match the original!
# Do NOT use DataFrameMapper for creating new columns, use a regular pipeline!
data_mapper = DataFrameMapper(
    tx_definitions,
input_df=True, df_out=True, default=None)
logging.info("Categorical transformer pipeline warnings, see docstring!".format())

# print("DataFrameMapper, applies transforms directly selected columns")
# for i, step in enumerate(data_mapper.features):
#     print(i, step)

#%% FIT TRANSFORM
df_all = data_mapper.fit_transform(df_all)
logging.info("Size of train df_all with categorical columns: {} MB".format(sys.getsizeof(df_all)/1000/1000))
#%% WARNING - sklearn-pandas has a flaw, it does not preserve categorical features!!!
for col in label_maps:
    # print(col)
    df_all[col] = df_all[col].astype('category')
logging.info("Reapplied categorical features".format())
logging.info("Size of df_all with categorical features: {} MB".format(sys.getsizeof(df_all)/1000/1000))


#%% SUMMARY

logging.info("Final shape of df_all {}".format(df_all.shape))
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
        ('feat: Pure Breed', trf.MultipleToNewFeature(['Breed1','Breed2'], 'Pure Breed', pure_breed)),
        ])

logging.info("Created pipeline:")
for i, step in enumerate(this_pipeline.steps):
    print(i, step[0], step[1].__str__())

#%% Fit Transform
original_cols = df_all.columns
df_all = this_pipeline.fit_transform(df_all)
logging.info("Pipeline complete. {} new columns.".format(len(df_all.columns) - len(original_cols)))

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


logging.info("Feature selection".format())
original_columns = df_all.columns
# col_selection = [col for col in all_columns if col not in cols_to_discard]

df_all.drop(cols_to_discard,inplace=True, axis=1)

logging.info("Selected {} of {} columns".format(len(df_all.columns),len(original_columns)))
logging.info("Size of df_all with selected features: {} MB".format(sys.getsizeof(df_all)/1000/1000))

logging.info("Record selection (sampling)".format())
logging.info("Sampling fraction: {}".format(SAMPLE_FRACTION))
df_all = df_all.sample(frac=SAMPLE_FRACTION)
logging.info("Final size of data frame: {}".format(df_all.shape))
logging.info("Size of df_all with selected features and records: {} MB".format(sys.getsizeof(df_all)/1000/1000))

# It is necessary to strictly remap the target variable!
target_col = 'AdoptionSpeed'
df_all[target_col]
inverse_map = {v: k for k, v in label_maps[target_col].items()}
df_all[target_col] = df_all[target_col].astype('object').replace(inverse_map)
df_all[target_col] = df_all[target_col].fillna(-1).astype('int64')#%%

df_tr = df_all[df_all['dataset_type']=='train'].copy()
df_tr.drop('dataset_type', axis=1, inplace=True)
logging.info("Split off train set {}, {:.1%} of the records".format(df_tr.shape,len(df_tr)/len(df_all)))

df_te = df_all[df_all['dataset_type']=='test'].copy()
df_te.drop('dataset_type', axis=1, inplace=True)
logging.info("Split off test set {}, {:.1%} of the records".format(df_tr.shape,len(df_te)/len(df_all)))

target_col = 'AdoptionSpeed'
y_tr = df_tr[target_col]
logging.info("Split off y_tr {}".format(len(y_tr)))

# Drop the target
X_tr = df_tr.drop(['AdoptionSpeed'], axis=1)
logging.info("Split off X_tr {}".format(X_tr.shape))

# Drop the target (it's NaN anyways)
X_te = df_te.drop(['AdoptionSpeed'], axis=1)
logging.info("Split off X_te {}".format(X_te.shape))

#%% DONE HERE - DELETE UNUSED

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

assert y_tr.dtype == np.dtype('int64'), "y_tr must be integer for LGBM!!"
#%% Model and params
params_model = dict()
# params['num_class'] = len(y_tr.value_counts())
params_model.update({
 'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': -1,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample_for_bin': 200000,
    'objective': 'multiclass',
 'class_weight': None,
    'min_split_gain': 0.0,
    'min_child_weight': 0.001,
    'min_child_samples': 20,
    'subsample': 1.0,
    'subsample_freq': 0,
 'colsample_bytree': 1.0,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'random_state': None,
    'n_jobs': -1, # -1 is for ALL
 'importance_type': 'split',
 'silent': True,
})
clf = lgb.LGBMClassifier(**params_model,
                         )

#%% GridCV
params_grid = {
    'learning_rate': [0.005, 0.05, 0.1, 0.2],
    # 'n_estimators': [40],
    # 'num_leaves': [6,8,12,16],
    # 'boosting_type' : ['gbdt'],
    # 'objective' : ['binary'],
    # 'random_state' : [501], # Updated from 'seed'
    # 'colsample_bytree' : [0.65, 0.66],
    # 'subsample' : [0.7,0.75],
    # 'reg_alpha' : [1,1.2],
    # 'reg_lambda' : [1,1.2,1.4],
    }

clf_grid = sk.model_selection.GridSearchCV(clf, params_grid,
                                       verbose=1,
                                       cv=5,
                                       n_jobs=-1)
#%% Fit
clf_grid.fit(X_tr, y_tr)

# Print the best parameters found
print("Best score:", clf_grid.best_score_)
print("Bast parameters:", clf_grid.best_params_)

clf_grid_BEST = clf_grid.best_estimator_

#%% Predict on X_tr for comparison
y_tr_predicted = clf_grid_BEST.predict(X_tr)

train_kappa = kappa(y_tr,y_tr_predicted)

logging.info("Metric on training set: {:0.3f}".format(train_kappa))

#%% Do the final fit on the BEST estimator
# start = datetime.datetime.now()
# predicted = clf_grid_BEST.fit(train_X, train_Y)
# logging.info("Elapsed H:m:s: {}".format(datetime.datetime.now()-start))

#%% Metric

# kappa(target, train_predictions)
# rmse(target, [r[0] for r in results['train']])
# submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': test_predictions})
# submission.head()


#%%
# n_fold = 5
# folds = sk.model_selection.StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=15)
#
# for fold_n, (train_indices, valid_indices) in enumerate(folds.split(X_tr, y_tr)):
#     logging.info("Fold {:<4} {:0.2f}|{:0.2f}% started {}".format(fold_n,
#                                                        100*len(train_indices)/len(X_tr),
#                                                        100*len(valid_indices)/len(X_tr),
#                                                                   time.ctime()))
#     gc.collect()
#     X_tr_fold, X_val_fold = X_tr.iloc[train_indices], X_tr.iloc[valid_indices]
#     y_tr_fold, y_val_fold = y_tr.iloc[train_indices], y_tr.iloc[valid_indices]
#
#     ds_tr_fold = lgb.Dataset(X_tr_fold, label=y_tr_fold)
#     ds_val_data = lgb.Dataset(X_val_fold, label=y_val_fold)
#
#
#     pprint(model.get_params())
#     logging.info("Model instantiated".format())
#
#
#     # model = lgb.train(params,
#     #                   ds_tr_fold,
#     #                   num_boost_round=2000,
#     #                   valid_sets=[ds_tr_fold, ds_val_data],
#     #                   verbose_eval=100,
#     #                   early_stopping_rounds=200)
#
#         grid = sk.model_selection.GridSearchCV(mdl, gridParams,
#                             verbose=0,
#                             cv=4,
#                             n_jobs=2)
#%% Predict on Test set
# NB we only want the defaulters column!
predicted = clf_grid_BEST.predict(X_te)


#%% Open the submission
# with zipfile.ZipFile(path_data / "test.zip").open("sample_submission.csv") as f:
#     df_submission = pd.read_csv(f, delimiter=',')
df_submission = pd.read_csv(path_data / 'test' / 'sample_submission.csv', delimiter=',')


#%% Collect predicitons
submission = pd.DataFrame({'PetID': df_submission.PetID, 'AdoptionSpeed': [int(i) for i in predicted]})
submission.head()

#%% Create csv
submission.to_csv('submission.csv', index=False)