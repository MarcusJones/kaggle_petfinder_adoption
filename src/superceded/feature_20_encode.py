#https://roamanalytics.com/2016/10/28/are-categorical-variables-getting-lost-in-your-random-forests/
#%%
# df_all['AdoptionSpeed'].fillna(-1)
# a[pd.isnull(a)]
if 0:
    import pandas.api.types as ptypes
    encoder_list = list()

    columns = df_all.columns.tolist()

    columns.remove(target_col)

    for col in columns:
        if ptypes.is_categorical_dtype(df_all[col]):
            encoder_list.append((col, sk.preprocessing.LabelEncoder()))

        elif ptypes.is_string_dtype(df_all[col]):
            # encoder_list.append((col,'STR?'))
            continue

        elif ptypes.is_bool_dtype(df_all[col]):
            encoder_list.append((col, sk.preprocessing.LabelEncoder()))

        elif ptypes.is_int64_dtype(df_all[col]):
            encoder_list.append((col, None))

        elif ptypes.is_float_dtype(df_all[col]):
            encoder_list.append((col, None))

        else:
            pass
            # print('Skip')

    logging.info("Encoder list: {}".format(len(encoder_list)))
    trf_cols = list()
    for enc in encoder_list:
        trf_cols.append(enc[0])

    skipped_cols = set(df_all.columns) - set(trf_cols)
    logging.info("Keep skipped columns unchanged: {}".format(skipped_cols))
    for col in skipped_cols:
        encoder_list.append((col, None))


    #%%
    # NB: The DataFrameMapper loses categorical features
    # Keep target variable aside!

    df_target = df_all[target_col].cat.codes

    data_mapper = DataFrameMapper(encoder_list, input_df=True, df_out=True)

    #%%

    df_all = data_mapper.fit_transform(df_all.copy())
    logging.info("Encoded df_all".format())

    logging.info("Re-applied the target column".format())
    df_all[target_col] = df_target

