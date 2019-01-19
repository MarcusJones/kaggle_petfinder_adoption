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

# sample = df_all.iloc[0:10][['Breed1','Breed2']]
df_all['Pure Breed'] = df_all.apply(pure_breed,axis=1)
df_all.columns
# For inspection:
# df_breeds = df_all[['Breed1','Breed2','Pure Breed']]
