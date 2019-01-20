import time



#%%
def pure_breed(row):
    # print(row)
    mixed_breed_keywords = ['domestic', 'tabby', 'mixed']

    # Mixed if labelled as such
    if row['Breed1'] == 'Mixed Breed':
        return False

    # Possible pure if no second breed
    elif row['Breed2'] == 'NA':
        # Reject domestic keywords
        if any([word in row['Breed1'].lower() for word in mixed_breed_keywords]):
            return False
        else:
            return True
    else:
        return False
r = pure_breed
#%%
class MultipleToNewFeature(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """
    """
    def __init__(self, selected_cols, new_col_name,func):
        self.selected_cols = selected_cols
        self.new_col_name = new_col_name
        self.func = func

    def fit(self, X, y=None):
        return self
    @timeit
    def transform(self, df, y=None):
        # print(df)
        start = time.time()
        df[self.new_col_name] = df.apply(self.func, axis=1)
        print(self.log, time.time() - start,"{}({}) -> ['{}']".format(self.func.__name__,self.selected_cols,self.new_col_name))
        return df

#%%

this_pipeline = sk.pipeline.Pipeline([
        ('counr', MultipleToNewFeature(['Breed1','Breed2'], 'Pure Breed', pure_breed)),
        ])
logging.info("Created pipeline:")
for i, step in enumerate(this_pipeline.steps):
    print(i, step[0], step[1].__str__())

#%%
df_s2 = this_pipeline.fit_transform(df_all)

#%%

# sample = df_all.iloc[0:10][['Breed1','Breed2']]
df_all['Pure Breed'] = df_all.apply(pure_breed,axis=1)
df_all['Pure Breed'] = df_all['Pure Breed'].astype('category')
df_all.columns
df_all.info()
# For inspection:
# df_breeds = df_all[['Breed1','Breed2','Pure Breed']]

#%%



#%%
# r = df_all.sample(10)[['Type']]
# len(r)
# r[:] = 1





this_pipeline = sk.pipeline.Pipeline([
        ('counr', WordCounter('Breed2', 'newcol')),
        ])

# data_mapper2 = DataFrameMapper(
#     (['Breed1', 'Breed2'], NumericalToCat(None)),
#     input_df=True, df_out=True, default=None)

logging.info("Created pipeline:")
for i, step in enumerate(this_pipeline.steps):
    print(i, step[0], step[1].__str__()[:60])

#%%
# transformer_def_list = [
#     (['Breed1', 'Breed2'], MultipleToNewFeature('Test', pure_breed)),
#     # (['Breed1', 'Breed2'], PureBreed()),
#
# ]
#
# transformer_def_list = [
#     (['Breed2'], WordCounter('Breed2', 'newcol')),
#     # (['Breed1', 'Breed2'], PureBreed()),
#
# ]
#
# data_mapper2 = DataFrameMapper(transformer_def_list, input_df=True, df_out=True, default=None)
df_s = df_all.sample(10)[['Breed1', 'Breed2', 'Type']]

this_pipeline = sk.pipeline.Pipeline([
        ('counr', WordCounter('Breed2', 'newcol')),
        ])

# data_mapper2 = DataFrameMapper(
#     (['Breed1', 'Breed2'], NumericalToCat(None)),
#     input_df=True, df_out=True, default=None)

logging.info("Created pipeline:")
for i, step in enumerate(this_pipeline.steps):
    print(i, step[0], step[1].__str__()[:60])

#%% FIT TRANSFORM
df_s2 = this_pipeline.fit_transform(df_s)

#%%


