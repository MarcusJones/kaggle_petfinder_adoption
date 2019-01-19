#%% ===========================================================================
# Standard imports
# =============================================================================
import os
import yaml
from pathlib import Path
import sys
import zipfile
from datetime import datetime

#%%
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
logging.info(f"Load")

# def load_zip
with zipfile.ZipFile(path_data / "train.zip").open("train.csv") as f:
    train = pd.read_csv(f,delimiter=',')
with zipfile.ZipFile(path_data / "test.zip").open("test.csv") as f:
    test = pd.read_csv(f,delimiter=',')
with zipfile.ZipFile(path_data / "test.zip").open("sample_submission.csv") as f:
    test = pd.read_csv(f,delimiter=',')

breeds = pd.read_csv(path_data / "breed_labels.csv")
colors = pd.read_csv(path_data / "color_labels.csv")
states = pd.read_csv(path_data / "state_labels.csv")

logging.debug("Loaded train {}".format(train.shape))
logging.debug("Loaded test {}".format(test.shape))

# Add a column to label the source of the data
train['dataset_type'] = 'train'
test['dataset_type'] = 'test'
logging.debug("Added dataset_type column for origin".format())
all_data = pd.concat([train, test],sort=False)

#%% Type
map_type = {1:"Dog", 2:"Cat"}
train['Type'] = train['Type'].astype('category')
train['Type'].cat.rename_categories(map_type)
#%% Adoption Speed (TARGET!)

sample = train.sample(1000).copy()
sample.info()
sample["Type"] = sample["Type"].astype('category')
sample.info()

r = sample.Type.cat.categories

sample.Type.cat.rename_categories(map_type)


#%% DONE HERE - DELETE UNUSED
print("******************************")

del_vars =[
        # "train",
        # "sfpd_head",
        # "sfpd_kag_all",
        # "sfpd_kag_head",
        # "df_summary",
        # "util_path",
        ]
cnt = 0
for name in dir():
    if name in del_vars:
        cnt+=1
        del globals()[name]
logging.info(f"Removed {cnt} variables from memory")
del cnt, name, del_vars




