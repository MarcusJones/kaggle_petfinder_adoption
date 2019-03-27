logging.info(" *** Step 2: Load data *** ".format())
#%%
# Load data
#

#%% ===========================================================================
# Data source and paths
# =============================================================================
path_data = Path(CONTROL_PARAMS['PATH_DATA_ROOT'], r"").expanduser()
assert path_data.exists(), "Data path does not exist: {}".format(path_data)
logging.info("Data path {}".format(path_data))

#%% ===========================================================================
# Load data
# =============================================================================
logging.info(f"Loading files into memory")

df_train = pd.read_csv(path_data / 'train'/ 'train.csv')
df_train.set_index(['PetID'],inplace=True)
df_test = pd.read_csv(path_data / 'test' / 'test.csv')
df_test.set_index(['PetID'],inplace=True)

breeds = pd.read_csv(path_data / "breed_labels.csv")
colors = pd.read_csv(path_data / "color_labels.csv")
states = pd.read_csv(path_data / "state_labels.csv")

logging.info("Loaded train {}".format(df_train.shape))
logging.info("Loaded test {}".format(df_test.shape))

target_col = 'AdoptionSpeed'
# Add a column to label the source of the data
df_train['dataset_type'] = 'train'
df_test['dataset_type'] = 'test'

# Set this aside for debugging
#TODO: Remove later
original_y_train = df_train['AdoptionSpeed'].copy()
original_y_train.value_counts()

logging.info("Added dataset_type column for origin".format())
df_all = pd.concat([df_train, df_test], sort=False)
# df_all.set_index('PetID',inplace=True)

del df_train, df_test

#%% Memory of the training DF:
logging.info("Size of df_all: {} MB".format(sys.getsizeof(df_all) / 1000 / 1000))
