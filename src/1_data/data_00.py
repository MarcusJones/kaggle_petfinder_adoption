#%% ===========================================================================
# Standard imports
# =============================================================================
import os
import yaml
import sys

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
#  Data source and paths
# =============================================================================
path_data = os.path.join(PATH_DATA_ROOT, r"")
assert os.path.exists(path_data), path_data
logging.info("Data path {}".format(PATH_DATA_ROOT))

#%% SFPD DATA
logging.info(f"Load SFPD")
sfpd_all = pd.read_csv(os.path.join(path_data, "rows.csv.gz"),delimiter=',',compression='gzip')
logging.info("Loaded SFPD data, {} rows".format(len(sfpd_all)))
#sfpd_head = sfpd_all.head()

#%% Create DateTime column on sfpd_all

#pd.options.mode.chained_assignment = None  # default='warn'
##sfpd_head["date_str"] = sfpd_head["Date"] + " " + sfpd_head["Time"]
#sfpd_all["dt"] = pd.to_datetime(sfpd_all.loc[:,"Date"] + " " + sfpd_all.loc[:,"Time"],format=r'%m/%d/%Y %H:%M')
#sfpd_all.drop(['DayOfWeek', 'Date', 'Time', 'Location'], axis=1, inplace=True)
#sfpd_head = sfpd_all.head()
#logging.info("Created dt column on SFPD data".format())

#%% Clean up the NULL entry
sfpd_all = sfpd_all[sfpd_all["PdDistrict"].isnull()==False]
# Assert that all columns are now the same length
df_summary = sfpd_all.describe(include = 'all')
assert df_summary.loc["count",:].astype(int).all()
logging.info("Dropped a null row!".format())
#logging.info("Loaded SFPD data, {} rows".format(len(sfpd_all)))

#%% Clean up the outliers (Remove)
total_rows = len(sfpd_all)
#long_outside = sum(sfpd_all.X < -121)
sfpd_all = sfpd_all.loc[sfpd_all.X < -121, :]
#dropped_long = total_rows - len(sfpd_all)
#total_rows= len(sfpd_all)
sfpd_all = sfpd_all.loc[sfpd_all.Y > -40, :]

dropped_rows = total_rows- len(sfpd_all)

#sum(sfpd_all["Y"]<40)
#sfpd_all = sfpd_all.dropna()
logging.info("Dropped {} outliers!".format(dropped_rows))

#%% KAGGLE DATA
if 0:
    logging.info(f"Load Kaggle SFPD")
    sfpd_kag_all = pd.read_csv(os.path.join(path_data, "Kaggle/train.csv.zip"),delimiter=',',compression='zip')
    logging.info("Loaded Kaggle SFPD data, {} rows".format(len(sfpd_kag_all)))
    sfpd_kag_head = sfpd_kag_all.head()

#%% Create DateTime column on sfpd_kag_all
#pd.options.mode.chained_assignment = None  # default='warn'
#sfpd_kag_all
#sfpd_kag_all["dt"] = pd.to_datetime(sfpd_kag_all.loc[:,"Dates"],format=r'%Y-%m-%d %H:%M:%S')
#sfpd_kag_all.drop(['DayOfWeek', 'Dates'], axis=1, inplace=True)
#sfpd_kag_head = sfpd_kag_all.head()
#logging.info("Created dt column on SFPD Kaggle data".format())


sfpd_head = sfpd_all.head()

#%% DONE HERE - DELETE UNUSED
print("******************************")

del_vars =[
        "path_data",
        "sfpd_head",
        "sfpd_kag_all",
        "sfpd_kag_head",
        "df_summary",
        "util_path",
        ]
cnt = 0
for name in dir():
    if name in del_vars:
        cnt+=1
        del globals()[name]
logging.info(f"Removed {cnt} variables from memory")
del cnt, name, del_vars




