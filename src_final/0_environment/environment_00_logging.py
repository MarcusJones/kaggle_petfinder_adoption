# %%
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 10:32:09 2018

@author: m.jones
"""

#%%
CONTROL_PARAMS = {
    # The machine running, either LOCAL or KAGGLE
    'DEPLOYMENT': None,

    # The type of run:
    # SIMPLE
    # KERNEL (Default)
    # SEARCH
    'RUN_TYPE' : None,
}


#%% ===========================================================================
# Logging
# =============================================================================
import sys
import logging
import datetime
import pprint

logger = logging.getLogger()
logger.handlers = []

# Set level
logger.setLevel(logging.INFO)

# Create formatter
FORMAT = "%(asctime)s : %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(FORMAT, DATE_FMT)

# Create handler and assign
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
logger.handlers = [handler]

CONTROL_PARAMS['START_TIME'] = datetime.datetime.now()
logging.info("Logging started")

logging.info("Kernel started {}".format(CONTROL_PARAMS['START_TIME']))
