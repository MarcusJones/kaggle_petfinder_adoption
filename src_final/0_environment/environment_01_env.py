import os
from pathlib import Path
# Set the environment

#%%
# Detect the deployment
if 'KAGGLE_WORKING_DIR' in os.environ:
    CONTROL_PARAMS['DEPLOYMENT'] = 'KAGGLE'
elif 'KERNEL_TEST' in os.environ:
    CONTROL_PARAMS['DEPLOYMENT'] = 'KERNEL_TEST'
else:
    CONTROL_PARAMS['DEPLOYMENT'] = 'LOCAL'

assert CONTROL_PARAMS['DEPLOYMENT'] in ['KAGGLE', 'LOCAL', 'KERNEL_TEST']

logging.info("Deployment: {}".format(CONTROL_PARAMS['DEPLOYMENT']))

#%%
# Build deployment params
if CONTROL_PARAMS['DEPLOYMENT'] =='KAGGLE':
    CONTROL_PARAMS['PATH_DATA_ROOT'] = Path.cwd() / '..' / 'input'
    CONTROL_PARAMS['SAMPLE_FRACTION'] = 1
    CONTROL_PARAMS['CV_FRACTION'] = 0

elif CONTROL_PARAMS['DEPLOYMENT']  == 'LOCAL':
    CONTROL_PARAMS['PATH_DATA_ROOT'] = Path.cwd() / 'input' / 'petfinder-adoption-prediction'
    # CONTROL_PARAMS['PATH_DATA_ROOT'] = r"~/DATA/petfinder_adoption"
    CONTROL_PARAMS['SAMPLE_FRACTION'] = 1
    CONTROL_PARAMS['CV_FRACTION'] = 0.2

elif CONTROL_PARAMS['DEPLOYMENT']  == 'KERNEL_TEST':
    CONTROL_PARAMS['PATH_DATA_ROOT'] = Path.cwd() / 'input' / 'petfinder-adoption-prediction'
    # CONTROL_PARAMS['PATH_DATA_ROOT'] = r"~/DATA/petfinder_adoption"
    CONTROL_PARAMS['SAMPLE_FRACTION'] = 0.5
    CONTROL_PARAMS['CV_FRACTION'] = 0.2

#%%
# Detect/set the run type
if CONTROL_PARAMS['DEPLOYMENT'] =='KAGGLE':
    CONTROL_PARAMS['RUN_TYPE']='SIMPLE'

elif CONTROL_PARAMS['DEPLOYMENT'] in ['LOCAL', 'KERNEL_TEST']:

    # Run a grid search, or a straight up training?
    CONTROL_PARAMS['RUN_TYPE']='SIMPLE'
    # CONTROL_PARAMS['RUN_TYPE']=='SEARCH'

    # Run a CV score?
    CONTROL_PARAMS['CV SCORE']=True
    CONTROL_PARAMS['CV FOLDS']=5
    # CONTROL_PARAMS['CV SCORE']=False

else:
    raise

logging.info("Run type: {}".format(CONTROL_PARAMS['RUN_TYPE']))

pprint.pprint(CONTROL_PARAMS)

#%% Build the run parameters
if CONTROL_PARAMS['RUN_TYPE']=='KFOLDS':
    CONTROL_PARAMS['CV FOLDS'] = 5