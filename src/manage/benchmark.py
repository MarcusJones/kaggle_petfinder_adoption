import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
import functools
from collections import OrderedDict

def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')
#%%
# Load the training data distribution



#%%
# Load reference kernel submissions
ref_submissions = dict()

name = 'Lukyanenko_Exploration_of_data_step_by_step 319'
ref_submissions[name] = Path.cwd() / 'references' / name / 'submission.csv'

name = 'Scuccimarra_PetFinder Simple LGBM Baseline 408'
ref_submissions[name] = Path.cwd() / 'references' / name / 'submission.csv'

dfs = dict()
for ref_key in ref_submissions:
    this_path = ref_submissions[ref_key]
    assert this_path.exists(), this_path
    dfs[ref_key] = pd.read_csv(this_path)
    dfs[ref_key].rename({'AdoptionSpeed':ref_key},inplace=True, axis='columns')
#%%
## df_final = functools.reduce(lambda left,right: pd.merge(left,right,on='PetID'), dfs)
df_list = [dfs[ref_key] for ref_key in dfs]

df_final = df_list.pop(0)
for df in df_list:
    print(df.columns)
    df_final = df_final.merge(df,on='PetID')

# df_final = dfs['one'].merge(dfs['two'],on='PetID')
df_final.set_index('PetID', drop=True, inplace=True)
df_final.describe()
# df_final.apply(pd.Series.value_counts, axis=1)
# %%
# count_df = pd.DataFrame()

for col in df_final:
    print(col)
    total = len(df_final[col])

    counts=df_final[col].value_counts()
    counts.name = 'Counts'

    percents = counts / total
    percents.name = 'Frequency'

    res = pd.concat([counts, percents], axis=1)
    res.sort_index(inplace=True)

    cnt_dict = counts.sort_index().to_dict()
    print(cnt_dict)


