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


# Set the environment
if 0:
    import os
    os.environ['LOCAL_KERNEL'] = '1'


# Detect the environment
if 'KAGGLE_WORKING_DIR' in os.environ:
    DEPLOYMENT = 'Kaggle'

elif 'LOCAL_KERNEL' in os.environ:
    DEPLOYMENT = 'Local Kernel'

else:
    DEPLOYMENT = 'Local'


logging.info("Deployment: {}".format(DEPLOYMENT))

# Set the environment
if DEPLOYMENT=='Kaggle':
    PATH_DATA_ROOT = Path.cwd() / '..' / 'input'
    SAMPLE_FRACTION = 1
    CV_FRACTION = 0
    RUN_TYPE = "Simple"
    FLAG_LOAD_TRANSFORMER = True
    RUN_TYPE = "Simple"

elif DEPLOYMENT == 'Local':
    PATH_DATA_ROOT = r"~/DATA/petfinder_adoption"
    SAMPLE_FRACTION = 1
    CV_FRACTION = 0.2
    FLAG_LOAD_TRANSFORMER = True
    RUN_TYPE = "Grid Search"

elif DEPLOYMENT == 'Local Kernel':
    PATH_DATA_ROOT = r"~/DATA/petfinder_adoption"
    SAMPLE_FRACTION = 0.5
    CV_FRACTION = 0.2
    FLAG_LOAD_TRANSFORMER = True
    RUN_TYPE = "Simple"


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
from functools import reduce
from collections import defaultdict

#%% ===========================================================================
#
# =============================================================================
# import coloredlogs
# coloredlogs.install()

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
import sklearn.ensemble
import sklearn.feature_extraction
import sklearn.decomposition
import sklearn.compose
import sklearn.utils
# import nltk
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from nltk.corpus import stopwords

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


