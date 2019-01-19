df_tr = df_all_final[df_all_final['dataset_type']=='train']
df_tr.drop('dataset_type',axis=1,inplace=True)
df_te = df_all_final[df_all_final['dataset_type']=='test']
df_te.drop('dataset_type',axis=1,inplace=True)

del df_all, df_all_final