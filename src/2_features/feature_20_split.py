#%%

df_tr = df_all[df_all['dataset_type']=='train'].copy()
df_tr.drop('dataset_type', axis=1, inplace=True)
logging.info("Split off train set {}, {:.1%} of the records".format(df_tr.shape,len(df_tr)/len(df_all)))

df_te = df_all[df_all['dataset_type']=='test'].copy()
df_te.drop('dataset_type', axis=1, inplace=True)
logging.info("Split off test set {}, {:.1%} of the records".format(df_tr.shape,len(df_te)/len(df_all)))

target_col = 'AdoptionSpeed'
y_tr = df_tr[target_col]
logging.info("Split off y_tr {}".format(len(y_tr)))

# Drop the target
X_tr = df_tr.drop(['AdoptionSpeed'], axis=1)
logging.info("Split off X_tr, dropped target {}".format(X_tr.shape))

# Drop the target (it's NaN anyways)
X_te = df_te.drop(['AdoptionSpeed'], axis=1)
logging.info("Split off X_te, dropped target {}".format(X_te.shape))

#%% DONE HERE - DELETE UNUSED

del_vars =[
    # 'df_all',
    # 'df_tr',
    # 'df_te',
]
cnt = 0
for name in dir():
    if name in del_vars:
        cnt+=1
        del globals()[name]
logging.info(f"Removed {cnt} variables from memory")
del cnt, name, del_vars

