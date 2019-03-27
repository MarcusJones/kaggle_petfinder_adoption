logging.info(" *** Step 4: Structure data ***".format())
#%%
# Instantiate and summarize
ds = DataStructure(df_all, target_col)
ds.train_test_summary()
ds.dtypes()

#%%
# Category counts
# ds.all_category_counts()
# ds.category_counts(target_col)

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
# Encode numeric
mapping_encoder = ds.build_encoder()
ds.apply_encoder(mapping_encoder)
ds.dtypes()


#%%
logging.info("Data Structure created: {}".format(ds))

#%%

# feature_design_space = ds.generate_design_space()
