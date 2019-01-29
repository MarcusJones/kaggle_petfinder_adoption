#https://roamanalytics.com/2016/10/28/are-categorical-variables-getting-lost-in-your-random-forests/
#%%
import pandas.api.types as ptypes
encoder_list = list()
for col in df_all.columns:
    if ptypes.is_categorical_dtype(df_all[col]):
        encoder_list.append((col,sk.preprocessing.LabelEncoder()))

    elif ptypes.is_string_dtype(df_all[col]):
        # encoder_list.append((col,'STR?'))
        continue

    elif ptypes.is_bool_dtype(df_all[col]):
        encoder_list.append((col, sk.preprocessing.LabelEncoder()))

    elif ptypes.is_bool_dtype(df_all[col]):
        encoder_list.append((col,sk.preprocessing.LabelEncoder()))

    elif ptypes.is_int64_dtype(df_all[col]):
        encoder_list.append((col,None))

    elif ptypes.is_float_dtype(df_all[col]):
        encoder_list.append((col,None))

    else:
        print('Skip')


trf_cols = list()
for enc in encoder_list:
    logging.info("{}".format(enc))
    trf_cols.append(enc[0])

skipped_cols = set(df_all.columns) - set(trf_cols)
# print(skipped_cols)
encoder_list.append(('dataset_type',None))
#%%
data_mapper = DataFrameMapper(encoder_list, input_df=True, df_out=True)
# ], input_df=True, df_out=True, default=None)

for step in data_mapper.features:
    print(step)
#%%
df_encoded = data_mapper.fit_transform(df_all.copy())
df_all = df_encoded
# df_trf_head = df_all_encoded.head()