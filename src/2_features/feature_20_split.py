#%%

df_tr = df_all[df_all['dataset_type']=='train'].copy()
df_tr.drop('dataset_type', axis=1, inplace=True)

df_te = df_all[df_all['dataset_type']=='test'].copy()
df_te.drop('dataset_type', axis=1, inplace=True)

y_tr = df_tr['AdoptionSpeed']
logging.info("y_tr {}".format(y_tr.shape))

X_tr = df_tr.drop(['AdoptionSpeed'], axis=1)
logging.info("X_tr {}".format(X_tr.shape))

X_te = df_te.drop(['AdoptionSpeed'], axis=1)
logging.info("X_te {}".format(X_te.shape))

#%% DONE HERE - DELETE UNUSED
print("******************************")

del_vars =[
    'df_all',
    'df_tr',
    'df_te',
]
cnt = 0
for name in dir():
    if name in del_vars:
        cnt+=1
        del globals()[name]
logging.info(f"Removed {cnt} variables from memory")
del cnt, name, del_vars

