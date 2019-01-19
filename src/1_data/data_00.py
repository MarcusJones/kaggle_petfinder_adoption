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

#%% Mapping types
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
label_maps['Breed1'] = label_maps
label_maps['Breed2'] = label_maps

# Similarly, load the color map
map_colors = dict(zip(colors['ColorID'], colors['ColorName']))
map_colors[0] = "NA"
label_maps['Color1'] = map_colors
label_maps['Color2'] = map_colors
label_maps['Color3'] = map_colors


#%% CATEGORICAL: Type
train['Type'] = train['Type'].astype('category')
train['Type'].cat.rename_categories(map_type, inplace=True)

#%% CATEGORICAL: Adoption Speed (TARGET!)

train['AdoptionSpeed'] = train['AdoptionSpeed'].astype('category')
train['AdoptionSpeed'].cat.rename_categories(map_adopt_speed,inplace=True)

#%% CATEGORICAL: Breeds


train['Breed1'] = train['Breed1'].astype('category')
train['Breed1'].cat.rename_categories(map_all_breeds,inplace=True)

train['Breed2'] = train['Breed2'].astype('category')
train['Breed2'].cat.rename_categories(map_all_breeds,inplace=True)

#%% CATEGORICAL: Gender


train['Gender'] = train['Gender'].astype('category')
train['Gender'].cat.rename_categories(map_gender,inplace=True)


#%% CATEGORICAL: MaturitySize
map_maturity_size = {
    1 : 'Small',
    2 : 'Medium',
    3 : 'Large',
    4 : 'Extra Large',
    0 : 'Not Specified',
}
train['MaturitySize'] = train['MaturitySize'].astype('category')
train['MaturitySize'].cat.rename_categories(map_maturity_size,inplace=True)

#%% CATEGORICAL: FurLength
map_FurLength = {
    1 : 'Short',
    2 : 'Medium',
    3 : 'Long',
    0 : 'Not Specified',
}

train['FurLength'] = train['FurLength'].astype('category')
train['FurLength'].cat.rename_categories(map_FurLength,inplace=True)


#%% Pipeline

data_mapper = DataFrameMapper([
    ('Vaccinated', NumericalToCat(label_maps),
], input_df=True, df_out=True, default=None)
# input_df - Ensure the passed in column enters as a series or DF
# df_out - Ensure the pipeline returns a df
# default - if a column is not transformed, keep it unchanged!

for step in data_mapper.features:
    print(step)

#%% FIT TRANSFORM
df_sample = train.sample(100).copy()
df_trf = data_mapper.fit_transform(df_sample)




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




