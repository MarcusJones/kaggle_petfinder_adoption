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
    df_train = pd.read_csv(f, delimiter=',')
with zipfile.ZipFile(path_data / "test.zip").open("test.csv") as f:
    df_test = pd.read_csv(f, delimiter=',')
with zipfile.ZipFile(path_data / "test.zip").open("sample_submission.csv") as f:
    df_sample_submission = pd.read_csv(f, delimiter=',')

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
logging.debug("Size of DF: {} MB".format(sys.getsizeof(df_all) / 1000 / 1000))

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
tx_definitions = [(col_name, NumericalToCat(label_maps)) for col_name in label_maps]

#%% Pipeline
# Build the pipeline
# NOTES:
# input_df - Ensure the passed in column enters as a series or DF
# df_out - Ensure the pipeline returns a df
# default - if a column is not transformed, keep it unchanged!
data_mapper = DataFrameMapper(
    tx_definitions,
input_df=True, df_out=True, default=None)

for step in data_mapper.features:
    print(step)

#%% FIT TRANSFORM
df_all = data_mapper.fit_transform(df_all)

logging.debug("Size of transformed train DF: {} MB".format(sys.getsizeof(df_all)/1000/1000))

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




