
class DataStructure:
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column

    def get_sub_df(self,dataset_type):
        sub_df = self.df[self.df['dataset_type'] == dataset_type]
        assert not sub_df._is_view
        return sub_df

    def sample_train(self, sample_frac):
        """Sample the training set to reduce size

        :return:
        """
        df_tr = self.get_sub_df('train')
        original_col_cnt = len(df_tr)
        df_te = self.get_sub_df('test')
        df_tr = df_tr.sample(frac=sample_frac)
        self.df = pd.concat([df_tr, df_te])
        logging.info("Sampled training set from {} to {} rows, fraction={:0.1%}".format(original_col_cnt, len(df_tr), len(df_tr)/original_col_cnt))

    def split_cv(self, cv_frac):
        df_tr, df_cv = sk.model_selection.train_test_split(df_tr, test_size=cv_frac)
        logging.info("Split off CV set, fraction={}".format(cv_frac))

    def split_train_test(self):
        df_tr = self.get_sub_df('train')
        y_tr = df_tr[self.target_column]
        X_tr = df_tr.drop([self.target_column], axis=1)

        df_te = self.get_sub_df('train')
        y_te = df_te[self.target_column]
        X_te = df_te.drop([self.target_column], axis=1)

        return (X_tr, y_tr, X_te, y_te)

    def train_test_summary(self):
        logging.info("DataFrame summary".format())
        logging.info("\tTarget column: {}".format(self.target_column))
        logging.info("\tTraining {}".format(self.get_sub_df('train').shape))
        logging.info("\tTest {}".format(self.get_sub_df('test').shape))
        logging.info("".format())

    def dtypes(self):
        dtype_dict = defaultdict(lambda: 0)
        for col in self.df.columns:
            dtype_dict[(str(self.df[col].dtype))] += 1
        logging.info("DataFrame dtypes:".format())
        for k in dtype_dict:
            logging.info("\t{:>10} : {}".format(k, dtype_dict[k]))

    def discard_features(self, col_list):
        logging.info("Discard columns".format())
        original_columns = self.df.columns
        discard_cols = [col for col in col_list if col in original_columns]
        self.df.drop(discard_cols, inplace=True, axis=1)
        if len(col_list) - len(discard_cols) > 0:
            logging.info("{} columns not found, ignoring".format(len(col_list) - len(discard_cols)))
        if len(discard_cols) > 0:
            logging.info("Discarded {} cols: {}".format(len(col_list), col_list))

    def encode_numeric(self):
        pass

#%%
# Instantiate and summarize
ds = DataStructure(df_all, 'AdoptionSpeed')
ds.train_test_summary()
ds.dtypes()

#%%
# Select feature columns
logging.info("Feature selection".format())
cols_to_discard = [
    'RescuerID',
    'Description',
    'Name',
]
ds.discard_features(cols_to_discard)


#%% Sample
df_all.columns
ds.sample_train(0.8)

X_tr, y_tr, X_te, y_te = ds.split_train_test()






#%%


df_te = df_all[df_all['dataset_type']=='test'].copy()
df_te.drop('dataset_type', axis=1, inplace=True)
logging.info("Split off test set {}, {:.1%} of the records".format(df_tr.shape,len(df_te)/len(df_all)))

logging.info("DataFrame summary".format())
logging.info("\tTraining {}".format(df_tr.shape))
if CV_FRACTION > 0:
    logging.info("\tCross Validation {}".format(df_cv.shape))
logging.info("\tTest {}".format(df_te.shape))

#%%
logging.info("Splitting into X_ and y_".format())
#%% Split Train

y_tr = df_tr[target_col]
X_tr = df_tr.drop(['AdoptionSpeed'], axis=1)

#%% Split CV
y_cv = df_cv[target_col]
X_cv = df_cv.drop(['AdoptionSpeed'], axis=1)
logging.info("Cross Validation X {}, y {}".format(X_cv.shape, y_cv.shape))

#%% Split Test
# Drop the target (it's NaN anyways)
X_te = df_te.drop(['AdoptionSpeed'], axis=1)

#%%
logging.info("X/y summary".format())


logging.info("\t{:0.1%} Training X {}, y {}".format(len(X_tr)/len(df_all), X_tr.shape, y_tr.shape))
if CV_FRACTION > 0:
    logging.info("\t{:0.1%} Cross Validation X {}, y {}".format(len(X_cv)/len(df_all), X_cv.shape, y_cv.shape))
logging.info("\t{:0.

#%%



# TODO: NB that this SHUFFLES the dataframe!
# df_all = df_all.sample(frac=SAMPLE_FRACTION)
logging.info("Final size of data frame: {}".format(df_all.shape))
logging.info("Size of df_all with selected features and records: {} MB".format(sys.getsizeof(df_all) / 1000 / 1000))

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
