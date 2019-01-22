
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

logging.debug("Loaded train {}".format(df_train.shape))
logging.debug("Loaded test {}".format(df_test.shape))

# Add a column to label the source of the data
df_train['dataset_type'] = 'train'
df_test['dataset_type'] = 'test'

# Set this aside for debugging
#TODO: Remove later
original_y_train = df_train['AdoptionSpeed'].copy()

logging.debug("Added dataset_type column for origin".format())
df_all = pd.concat([df_train, df_test], sort=False)
# df_all.set_index('PetID',inplace=True)

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

#%% Restructure the dict
#%% DEBUG TRF

class TransformerLog():
    """Add a .log attribute for logging
    """
    @property
    def log(self):
        return "Transformer: {}".format(type(self).__name__)
class NumericalToCat(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """Convert numeric indexed column into dtype category with labels
    Convert a column which has a category, presented as an Integer
    Initialize with a dict of ALL mappings for this session, keyed by column name
    (This could be easily refactored to have only the required mapping)
    """
    def __init__(self,label_map):
        self.label_map = label_map

    def fit(self, X, y=None):
        return self

    def transform(self, this_series):
        assert type(this_series) == pd.Series
        mapped_labels = list(self.label_map.values())
        # assert this_series.name in self.label_map_dict, "{} not in label map!".format(this_series.Name)
        return_series = this_series.copy()
        return_series = pd.Series(pd.Categorical.from_codes(this_series, mapped_labels))
        # return_series = return_series.astype('category')
        # return_series.cat.rename_categories(self.label_map_dict[return_series.name], inplace=True)
        print(self.log, mapped_labels, return_series.cat.categories, )
        assert return_series.dtype == 'category'
        return return_series

this_series = df_all['Vaccinated'].copy()
this_series.value_counts()
label_map = label_maps['Vaccinated']
mapped_labels = list(label_map.values())
my_labels = pd.Index(mapped_labels)
pd.Series(pd.Categorical.from_codes(this_series, my_labels))

#%% Dynamically create the transformation definitions
tx_definitions = [(col_name, NumericalToCat(label_maps[col_name])) for col_name in label_maps]

tx_definitions = [tx_definitions[0]]

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
