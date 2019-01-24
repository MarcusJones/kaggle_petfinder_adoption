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
logger.setLevel(logging.info)

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
    SAMPLE_FRACTION = 1
    import kaggle_utils.transformers as trf


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
from catboost import CatBoostClassifier
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


