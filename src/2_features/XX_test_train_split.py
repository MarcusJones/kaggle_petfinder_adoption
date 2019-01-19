df_all_sample = df_all_final.sample(frac=0.3)


df_tr = df_all_sample[df_all_sample['dataset_type']=='train'].copy()
df_tr.drop('dataset_type',axis=1,inplace=True)
df_te = df_all_sample[df_all_sample['dataset_type']=='test'].copy()
df_te.drop('dataset_type',axis=1,inplace=True)

# del df_all, df_all_final


y_tr = df_tr['AdoptionSpeed']
X_tr = df_tr.drop(['AdoptionSpeed'], axis=1)

X_te = df_te.drop(['AdoptionSpeed'], axis=1)

X_tr.info()