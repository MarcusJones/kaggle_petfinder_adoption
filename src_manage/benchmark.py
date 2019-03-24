import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
import functools
from collections import OrderedDict
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')
#%%
# Load the training data distribution


#%%
# Load reference kernel submissions
ref_submissions = dict()

name = 'Lukyanenko_Exploration_of_data_step_by_step 319'
ref_submissions[name] = Path.cwd() / 'reference_kernels' / name / 'submission.csv'

name = 'Scuccimarra_PetFinder Simple LGBM Baseline 408'
ref_submissions[name] = Path.cwd() / 'reference_kernels' / name / 'submission.csv'

name = 'current_kernel'
ref_submissions[name] = Path.cwd() / 'kernel_submission' / 'submission.csv'

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
df_res = pd.DataFrame(index=range(5))
for col in df_final:
    print(col)
    total = len(df_final[col])

    counts=df_final[col].value_counts()
    counts.name = 'Counts'

    percents = counts / total
    percents.name = 'Frequency'

    res = pd.concat([counts, percents], axis=1)
    res.sort_index(inplace=True)
    df_res[col] = counts
    cnt_dict = counts.sort_index().to_dict()
    # print(sum(cnt_dict.values()), cnt_dict)

df_res.sum(axis=0)
this_barchart = df_res.plot.bar()
plt.show()

#%%
cnt_dict
bar = plt.bar(range(len(cnt_dict)), cnt_dict.values())
plt.xticks(range(len(cnt_dict)),)
plt.show()
fig, ax = plt.subplots()
rects1 = ax.bar(ind - width/2, men_means, width, yerr=men_std,
                color='SkyBlue', label='Men')
rects2 = ax.bar(ind + width/2, women_means, width, yerr=women_std,
                color='IndianRed', label='Women')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(ind)
ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
ax.legend()

