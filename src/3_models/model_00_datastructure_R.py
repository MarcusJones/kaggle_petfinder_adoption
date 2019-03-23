#%%
# Instantiate and summarize
ds = DataStructure(df_all, target_col)
ds.train_test_summary()
ds.dtypes()

#%% Sample
df_all.columns
ds.sample_train(control_dict['sample fraction'])


#%%
# Discard
# Select feature columns
logging.info("Feature selection".format())
cols_to_discard = [
    'RescuerID',
    'Description',
    'Name',
]
ds.discard_features(cols_to_discard)
ds.dtypes()

#%%
feature_cols = [col for col in control_dict['genome']['feature set']]
discard_from_genome = list()
for col in feature_cols:
    if not col['value']:
        discard_from_genome.append(col['name'])
ds.discard_features(discard_from_genome)

#%%
# Encode numeric
mapping_encoder = ds.build_encoder()
ds.apply_encoder(mapping_encoder)
ds.dtypes()

