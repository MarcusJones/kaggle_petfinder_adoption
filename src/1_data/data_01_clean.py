#%%
# Preprocess df_all
# - Convert all to categorical using provided labels
# DataFrameMapper is used
# EXTRA handling since mapper loses categorical
# EXTRA handling to ensure the target matches the original numerical labels

#%%
df_all['PhotoAmt'] = df_all['PhotoAmt'].astype('int')
# df_all['AdoptionSpeed'] = df_all['AdoptionSpeed'].fillna(-1)
# df_all['AdoptionSpeed'] = df_all['AdoptionSpeed'].astype('int')

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
    # -1 : "Empty",
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

logging.info("Category mappings for {} columns created".format(len(label_maps)))

for map in label_maps:
    print(map, label_maps[map])

#%% Dynamically create the transformation definitions
# tx_definitions_preview = [(col_name, label_maps[col_name]) for col_name in label_maps]
# for t in tx_definitions_preview:
#     print(t)
tx_definitions = [(col_name, trf.NumericalToCat(label_maps[col_name], True)) for col_name in label_maps]
# col_name = 'Vaccinated'
#%% Pipeline
# Build the pipeline
# NOTES:
# input_df - Ensure the passed in column enters as a series or DF
# df_out - Ensure the pipeline returns a df
# default - if a column is not transformed, keep it unchanged!
# WARNINGS:
# The categorical dtype is LOST!
# The mapping does NOT match the original!
# Do NOT use DataFrameMapper for creating new columns, use a regular pipeline!
data_mapper = DataFrameMapper(
    tx_definitions,
input_df=True, df_out=True, default=None)
logging.info("Categorical transformer pipeline warnings, see docstring!".format())

# print("DataFrameMapper, applies transforms directly selected columns")
# for i, step in enumerate(data_mapper.features):
#     print(i, step)

#%% FIT TRANSFORM
df_all = data_mapper.fit_transform(df_all)
logging.info("Size of train df_all with categorical columns: {} MB".format(sys.getsizeof(df_all)/1000/1000))
logging.info("Warning: Pipeline returns strings, not categorical! ".format(sys.getsizeof(df_all)/1000/1000))

#%% WARNING - sklearn-pandas has a flaw, it does not preserve categorical features!!!
# ordered = ['AdoptionSpeed','MaturitySize','FurLength','Health'] #Actually, most have 'unspecified', which can't be ordered
ordered_cols = ['AdoptionSpeed']
for col in label_maps:
    this_labels = sorted(list(label_maps[col].items()), key=lambda tup: tup[0])
    listed_categories = [tup[1] for tup in this_labels]
    if col in listed_categories:
        ordered_flag = True
    else:
        ordered_flag = False
    cat_type = pd.api.types.CategoricalDtype(categories=listed_categories,ordered=True)
    df_all[col] = df_all[col].astype(cat_type)

logging.info("Reapplied categorical features".format())
logging.info("Size of df_all with categorical features: {} MB".format(sys.getsizeof(df_all)/1000/1000))

#%%
# df_all['AdoptionSpeed'].value_counts()
# ser = df_all['AdoptionSpeed']
#
# original_y_train.value_counts()


#%% SUMMARY

logging.info("Final shape of df_all {}".format(df_all.shape))
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
