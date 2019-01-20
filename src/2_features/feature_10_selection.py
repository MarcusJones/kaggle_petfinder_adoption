#%%
# The final selection of columns from the main DF
cols_to_use = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength',
               'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'RescuerID', 'VideoAmt',
               'PhotoAmt', 'AdoptionSpeed', 'No_name', 'Pure_breed', 'health', 'Free',
               'score', 'magnitude']

cols_to_discard = [
    'RescuerID',
    'Description',
    'Name',
]


logging.debug("".format())
all_columns = df_all.columns
col_selection = [col for col in all_columns if col not in cols_to_discard]

df_all.drop(col_selection,inplace=True, axis=1)

logging.debug("Selected {} of {} columns".format(len(col_selection),len( all_columns )))
logging.debug("Size of df_all with selected features: {} MB".format(sys.getsizeof(df_all)/1000/1000))