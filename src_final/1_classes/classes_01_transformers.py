logging.info(" *** Step 1: Classes ***".format())
#%%
# This is to work around kaggle kernel's not allowing external modules
# For local deployment, this is skipped.
# For kaggle kernel deployment, the transformers and utilities are loaded.

#%%
def timeit(method):
    """ Decorator to time execution of transformers
    :param method:
    :return:
    """

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print("\t {} {:2.1f}s".format(method.__name__, (te - ts)))
        return result

    return timed

class TransformerLog():
    """Add a .log attribute for logging
    """

    @property
    def log(self):
        return "Transformer: {}".format(type(self).__name__)

#%%
class MultipleToNewFeature(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """Given a list of column names, create a new column in the df.

    """

    def __init__(self, selected_cols, new_col_name, func):
        self.selected_cols = selected_cols
        self.new_col_name = new_col_name
        self.func = func

    def fit(self, X, y=None):
        return self

    @timeit
    def transform(self, df, y=None):
        df[self.new_col_name] = df.apply(self.func, axis=1)
        print(self.log, "{}({}) -> ['{}']".format(self.func.__name__, self.selected_cols, self.new_col_name))
        return df

#%%
class NumericalToCat(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """Convert numeric indexed column into dtype category with labels
    Convert a column which has a category, presented as an Integer
    Initialize with a dict of ALL mappings for this session, keyed by column name
    (This could be easily refactored to have only the required mapping)
    """

    def __init__(self, label_map_dict, allow_more_labels=False):
        self.label_map_dict = label_map_dict
        self.allow_more_labels = allow_more_labels

    def fit(self, X, y=None):
        return self

    def get_unique_values(self, this_series):
        return list(this_series.value_counts().index)

    def transform(self, this_series):
        if not self.allow_more_labels:
            if len(self.label_map_dict) > len(this_series.value_counts()):
                msg = "{} labels provided, but {} values in column!\nLabels:{}\nValues:{}".format(
                    len(self.label_map_dict), len(this_series.value_counts()), self.label_map_dict,
                    self.get_unique_values(this_series), )
                raise ValueError(msg)

        if len(self.label_map_dict) < len(this_series.value_counts()):
            raise ValueError

        assert type(this_series) == pd.Series
        # assert this_series.name in self.label_map_dict, "{} not in label map!".format(this_series.name)
        return_series = this_series.copy()
        # return_series = pd.Series(pd.Categorical.from_codes(this_series, self.label_map_dict))
        return_series = return_series.astype('category')
        return_series.cat.rename_categories(self.label_map_dict, inplace=True)
        # print(return_series.cat.categories)

        assert return_series.dtype == 'category'
        return return_series

# Here we simulate a module namespace
class trf:
    NumericalToCat = NumericalToCat
    MultipleToNewFeature = MultipleToNewFeature
    TransformerLog = TransformerLog
    timeit = timeit


#%%
class PandasSelector(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    def __init__(self, columns=None, dtype=None, inverse=False,
                 return_vector=True, name=None):
        self.dtype = dtype
        self.columns = columns
        self.inverse = inverse
        self.return_vector = return_vector
        self.name = name

        # Ensure columns in a list()
        if isinstance(self.columns, str):
            self.columns = [self.columns]

        logging.info("Init {} on cols: {}".format(name, columns))

    def check_condition(self, x, col):
        cond = (self.dtype is not None and x[col].dtype == self.dtype) or \
               (self.columns is not None and col in self.columns)
        return self.inverse ^ cond

    def fit(self, x, y=None):
        return self

    def _check_if_all_columns_present(self, x):
        if not self.inverse and self.columns is not None:
            missing_columns = set(self.columns) - set(x.columns)
            if len(missing_columns) > 0:
                missing_columns_ = ','.join(col for col in missing_columns)
                raise KeyError(
                    "Keys are missing in the record: {}, columns required:{}".format(missing_columns_, self.columns))

    def transform(self, x):
        logging.info("{} is transforming...".format(self.name))
        # check if x is a pandas DataFrame
        if not isinstance(x, pd.DataFrame):
            raise KeyError("Input is not a pandas DataFrame it's a {}".format(type(x)))

        selected_cols = []
        for col in x.columns:
            if self.check_condition(x, col):
                selected_cols.append(col)

        # if the column was selected and inversed = False make sure the column
        # is in the DataFrame
        self._check_if_all_columns_present(x)

        # if only 1 column is returned return a vector instead of a dataframe
        if len(selected_cols) == 1 and self.return_vector:
            return list(x[selected_cols[0]])
        else:
            return x[selected_cols]

#%%
class PandasSelector2(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    def __init__(self, columns=None, dtype=None, inverse=False,
                 name=None):
        self.dtype = dtype
        self.columns = columns
        self.inverse = inverse
        # self.return_vector = return_vector
        self.name = name

        # Ensure columns in a list()
        if isinstance(self.columns, str):
            self.columns = [self.columns]

        logging.info("Init {} on cols: {}".format(name, columns))

    def check_condition(self, x, col):
        cond = (self.dtype is not None and x[col].dtype == self.dtype) or \
               (self.columns is not None and col in self.columns)
        return self.inverse ^ cond

    def fit(self, x, y=None):
        return self

    def _check_if_all_columns_present(self, x):
        if not self.inverse and self.columns is not None:
            missing_columns = set(self.columns) - set(x.columns)
            if len(missing_columns) > 0:
                missing_columns_ = ','.join(col for col in missing_columns)
                raise KeyError(
                    "Keys are missing in the record: {}, columns required:{}".format(missing_columns_, self.columns))

    def transform(self, df):
        logging.info("{} is transforming...".format(self.name))
        # check if pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise KeyError("Input is not a pandas DataFrame it's a {}".format(type(df)))

        selected_cols = []
        for col in df.columns:
            if self.check_condition(df, col):
                selected_cols.append(col)

        # if the column was selected and inversed = False make sure the column is in the DataFrame
        self._check_if_all_columns_present(df)

        # if only 1 column is returned return a vector instead of a dataframe
        return df[selected_cols]
