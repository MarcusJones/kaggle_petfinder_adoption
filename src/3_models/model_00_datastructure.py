
class DataStructure:
    def __init__(self, df, target_column):
        self.df = df.copy()
        self.target_column = target_column
        self.dataset_type_column = 'dataset_type'
        self.feature_columns = set(self.df.columns) - set([self.target_column, self.dataset_type_column])

        logging.info("Dataset with {} features, {} records".format(len(self.feature_columns), len(self.df)))

    def get_sub_df(self,dataset_type):
        sub_df = self.df[self.df[self.dataset_type_column] == dataset_type]
        assert not sub_df._is_view
        # sub_df.drop(dataset_type)
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
        logging.info("\tDataset type column: {}".format(self.dataset_type_column))
        len_all = len(self.df)
        len_tr = len(self.get_sub_df('train'))
        len_te = len(self.get_sub_df('test'))
        logging.info("\tTraining {:<8} {:0.1%}".format(len_tr, len_tr/len_all))
        logging.info("\t    Test {:<8} {:0.1%}".format(len_te, len_te/len_all))

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

    def all_category_counts(self):
        for col in self.df.columns:
            if pd.api.types.is_categorical_dtype(self.df[col]):
                self.category_counts(col)

    def category_counts(self, col_name):
        ds.df[col_name].cat.categories
        logging.info("{} category counts".format(col_name))
        for cat, (label, count) in enumerate(ds.df[col_name].value_counts().iteritems()):
            logging.info("\t{:5} = {:30} {}".format(cat, label, count))

    def build_encoder(self):
        encoder_list = list()

        columns = self.df.columns.tolist()

        columns.remove(self.target_column)
        columns.remove(self.dataset_type_column)

        for col in columns:
            if pd.api.types.is_categorical_dtype(self.df[col]):
                encoder_list.append((col, sk.preprocessing.LabelEncoder()))

            elif pd.api.types.is_string_dtype(self.df[col]):
                # encoder_list.append((col,'STR?'))
                continue

            elif pd.api.types.is_bool_dtype(self.df[col]):
                encoder_list.append((col, sk.preprocessing.LabelEncoder()))

            elif pd.api.types.is_int64_dtype(self.df[col]):
                encoder_list.append((col, None))

            elif pd.api.types.is_float_dtype(self.df[col]):
                encoder_list.append((col, None))

            else:
                pass

        logging.info("Encoder list: {}".format(len(encoder_list)))
        trf_cols = list()
        for enc in encoder_list:
            trf_cols.append(enc[0])

        skipped_cols = set(self.df.columns) - set(trf_cols)
        logging.info("Keep skipped columns unchanged: {}".format(skipped_cols))
        for col in skipped_cols:
            encoder_list.append((col, None))

        # df_target = df_all[target_col].cat.codes

        data_mapper = DataFrameMapper(encoder_list, input_df=True, df_out=True)

        return data_mapper

    @property
    def ready_to_split(self):
        return False

    # def target_cat_to_numeric(self):
    #     self.df[self.target_column] = self.df[self.target_column].cat.codes

    def assert_all_numeric(self):
        skip = [self.dataset_type_column, self.target_column]
        # numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        # newdf = df.select_dtypes(include=numerics)
        for col in self.df:
            if col in skip:
                continue
            assert pd.api.types.is_numeric_dtype(self.df[col]), "{} not numeric".format(col)


    def apply_encoder(self, encoder, target_handler=None):

        if not target_handler:
            df_target = self.df[self.target_column].cat.codes
            logging.info("Target will be mapped back to category numbers".format())

        this_df = encoder.fit_transform(self.df.copy())
        logging.info("Encoded df".format())

        # Add target column
        this_df[self.target_column] = df_target

        # Add the dataset type column
        logging.info("".format())
        this_df[self.dataset_type_column] = self.df[self.dataset_type_column]
        self.df = this_df
        self.assert_all_numeric()




#%%
# Instantiate and summarize
ds = DataStructure(df_all, target_col)
ds.train_test_summary()
ds.dtypes()


#%%
# Category counts
# ds.all_category_counts()
# ds.category_counts(target_col)

#%% Sample
df_all.columns
ds.sample_train(0.8)


#%%
# Discard
# Select feature columns
logging.info("Feature selection".format())
cols_to_discard = [
    'RescuerID',
    'Description',
    'Name',
]
ds.discard_features(cols_to_discard)
ds.dtypes()

#%%
# Encode numeric
mapping_encoder = ds.build_encoder()
ds.apply_encoder(mapping_encoder)
ds.dtypes()


#%%
# Split
X_tr, y_tr, X_te, y_te = ds.split_train_test()

