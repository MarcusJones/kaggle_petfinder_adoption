# It is necessary to strictly remap the target variable!
target_col = 'AdoptionSpeed'
df_all[target_col]
inverse_map = {v: k for k, v in label_maps[target_col].items()}
df_all[target_col] = df_all[target_col].astype('object').replace(inverse_map)
df_all[target_col] = df_all[target_col].fillna(-1).astype('int64')