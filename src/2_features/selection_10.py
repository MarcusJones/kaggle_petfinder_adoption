#%%
cols_to_use = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength',
               'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'RescuerID', 'VideoAmt',
               'PhotoAmt', 'AdoptionSpeed', 'No_name', 'Pure_breed', 'health', 'Free',
               'score', 'magnitude']

cols_to_discard = [
    'RescuerID',
    'Description',
    'Name',
]
df_all_final = df_all[[col for col in df_all.columns if col not in cols_to_discard]]

df_all_final
