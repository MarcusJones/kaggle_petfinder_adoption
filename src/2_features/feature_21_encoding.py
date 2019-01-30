#https://roamanalytics.com/2016/10/28/are-categorical-variables-getting-lost-in-your-random-forests/
#%%
# df_all['AdoptionSpeed'].fillna(-1)
# a[pd.isnull(a)]
import pandas.api.types as ptypes
encoder_list = list()
for col in X_tr.columns:
    if ptypes.is_categorical_dtype(X_tr[col]):
        encoder_list.append((col,sk.preprocessing.LabelEncoder()))

    elif ptypes.is_string_dtype(X_tr[col]):
        # encoder_list.append((col,'STR?'))
        continue

    elif ptypes.is_bool_dtype(X_tr[col]):
        encoder_list.append((col, sk.preprocessing.LabelEncoder()))

    elif ptypes.is_bool_dtype(X_tr[col]):
        encoder_list.append((col,sk.preprocessing.LabelEncoder()))

    elif ptypes.is_int64_dtype(X_tr[col]):
        encoder_list.append((col,None))

    elif ptypes.is_float_dtype(X_tr[col]):
        encoder_list.append((col,None))

    else:
        print('Skip')


trf_cols = list()
for enc in encoder_list:
    logging.info("{}".format(enc))
    trf_cols.append(enc[0])

skipped_cols = set(X_tr.columns) - set(trf_cols)
# print(skipped_cols)
# encoder_list.append(('dataset_type',None))
#%%
data_mapper = DataFrameMapper(encoder_list, input_df=True, df_out=True)
# ], input_df=True, df_out=True, default=None)

for step in data_mapper.features:
    print(step)
#%%
X_tr = data_mapper.fit_transform(X_tr.copy())
X_te = data_mapper.fit_transform(X_te.copy())
logging.info("Encoded X_tr and X_te".format())
y_tr = y_tr.cat.codes
logging.info("Reverted target to integers".format())
# df_trf_head = df_all_encoded.head()