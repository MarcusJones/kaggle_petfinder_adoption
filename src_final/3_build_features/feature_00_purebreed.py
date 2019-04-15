logging.info(" *** Step 3: Build features *** ".format())

# %% ===========================================================================
# Feature - Pure Breed Boolean column
# =============================================================================
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

#%% Build the pipeline
this_pipeline = sk.pipeline.Pipeline([
        ('feat: Pure Breed', trf.MultipleToNewFeature(['Breed1','Breed2'], 'Pure Breed', pure_breed)),
        ])

logging.info("Created pipeline:")
for i, step in enumerate(this_pipeline.steps):
    print(i, step[0], step[1].__str__())

#%% Fit Transform
original_cols = df_all.columns
df_all = this_pipeline.fit_transform(df_all)
logging.info("Pipeline complete. {} new columns.".format(len(df_all.columns) - len(original_cols)))

