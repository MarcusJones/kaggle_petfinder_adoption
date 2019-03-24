class DataStructure:
    def __init__(self, df, target_column, dataset_type_column='dataset_type'):
        self.df = df.copy()
        self.dataset_type_column = dataset_type_column
        self.target_column = target_column
        self.feature_columns = set(self.df.columns) - set([self.target_column, self.dataset_type_column])

        logging.info("Dataset with {} features, {} records".format(len(self.feature_columns), len(self.df)))

    @property
    def search_grid(self):
        search_grid = list()
        for col in self.feature_columns:
            search_grid.append({'name': col, 'vtype': 'bool', 'variable_tuple':[True, False], 'ordered':False})
        return search_grid

    def get_dataset_type_df(self, dataset_type):
        sub_df = self.df[self.df[self.dataset_type_column] == dataset_type]
        assert not sub_df._is_view
        # sub_df.drop(dataset_type)
        return sub_df

    def sample_train(self, sample_frac):
        """Sample the training set to reduce size
        Do not sample the test records

        :return:
        """
        df_tr = self.get_dataset_type_df('train')
        original_col_cnt = len(df_tr)
        # Set the test records aside
        df_te = self.get_dataset_type_df('test')
        df_tr = df_tr.sample(frac=sample_frac)
        self.df = pd.concat([df_tr, df_te])
        logging.info("Sampled training set from {} to {} rows, fraction={:0.1%}".format(original_col_cnt, len(df_tr), len(df_tr)/original_col_cnt))

    def split_train_test(self):
        df_tr = self.get_dataset_type_df('train')
        y_tr = df_tr[self.target_column]
        X_tr = df_tr.drop([self.target_column, self.dataset_type_column], axis=1)

        df_te = self.get_dataset_type_df('train')
        y_te = df_te[self.target_column]
        X_te = df_te.drop([self.target_column, self.dataset_type_column], axis=1)

        return (X_tr, y_tr, X_te, y_te)

    def train_test_summary(self):
        logging.info("DataFrame summary".format())
        logging.info("\tTarget column: {}".format(self.target_column))
        logging.info("\tDataset type column: {}".format(self.dataset_type_column))
        len_all = len(self.df)
        len_tr = len(self.get_dataset_type_df('train'))
        len_te = len(self.get_dataset_type_df('test'))
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

        for col in col_list:
            self.feature_columns.remove(col)

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




if 0:
    #%%
    @depcretiated
    def cross_validate(_X_tr, _y_tr, model, metric_function, folds=10, repeats=5):
        """
        Function to do the cross validation - using stacked Out of Bag method instead of averaging across folds.
        model = algorithm to validate. Must be scikit learn or scikit-learn like API (Example xgboost XGBRegressor)
        x = training data, numpy array
        y = training labels, numpy array
        folds = K, the number of folds to divide the data into
        repeats = Number of times to repeat validation process for more confidence
        :param _X_tr:
        :param _y_tr:
        :param model: SKLearn API Model
        :param metric_function: takes y, y_pred
        :param folds:
        :param repeats:
        :return:
        """
        y_cv_fold_pred = np.zeros((len(_y_tr), repeats))
        score = np.zeros(repeats)
        _X_tr = np.array(_X_tr)
        for r in range(repeats):
            i=0
            logging.info("Cross Validating - Run {} of {}".format(str(r + 1), str(repeats)))
            _X_tr, _y_tr = sk.utils.shuffle(_X_tr, _y_tr, random_state=r)    #shuffle data before each repeat
            kf = sk.model_selection.KFold(n_splits=folds, random_state=i+1000)         #random split, different each time
            for train_ind, cv_ind in kf.split(_X_tr):
                print('Fold', i+1, 'out of', folds)
                _X_tr_fold, _y_tr_fold = _X_tr[train_ind, :], _y_tr[train_ind]
                _X_cv_fold, _y_cv_fold = _X_tr[cv_ind, :], _y_tr[cv_ind]
                model.fit(_X_tr_fold, _y_tr_fold)
                y_cv_fold_pred[cv_ind, r] = model.predict(_X_cv_fold)
                i+=1
            score[r] = metric_function(y_cv_fold_pred[:,r], _y_tr)

        logging.info('\nCV complete, overall score: {}'.format(str(score)))
        logging.info('Mean: {}'.format(str(np.mean(score))))
        logging.info('Deviation: {}'.format(str(np.std(score))))

    #%%
    """
    CV complete, overall score: [0.31460838 0.31986689 0.31859581 0.32246347 0.32236281]
    2019-03-09 22:01:39 : Mean: 0.31957947269358394
    2019-03-09 22:01:39 : Deviation: 0.002892279348388643
    """
    cross_validate(X_tr, y_tr, clf, kappa)


