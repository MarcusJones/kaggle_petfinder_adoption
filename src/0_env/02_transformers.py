import sklearn as sk
import sklearn.preprocessing
import pandas as pd
import time
import numpy as np
from sklearn_pandas import DataFrameMapper


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
#        if 'log_time' in kw:
#            name = kw.get('log_name', method.__name__.upper())
#            kw['log_time'][name] = int((te - ts) * 1000)
        if 0:
            pass
        else:
            pass
#            print('Elapsed', (te - ts) * 1000)
            #print '%r  %2.2f ms' % \
            #      (method.__name__, (te - ts) * 1000)
        return result
    return timed

class TransformerLog():
    """Add a .log attribute for logging
    """
    @property
    def log(self):
        return "Transformer: {}".format(type(self).__name__)

# %%==============================================================================
# Empty
# ===============================================================================
class Imputer1D(sk.preprocessing.Imputer):
    """
    A simple wrapper class on Imputer to avoid having to make a single column 2D.
    """
    def fit(self, X, y=None):
        if X.ndim == 1:
            X = np.expand_dims(X, axis=1)
        # Call the Imputer as normal, return result
        return super(Imputer1D, self).fit(X, y=None)

    def transform(self, X, y=None):
        if X.ndim == 1:
            X = np.expand_dims(X, axis=1)
            # Call the Imputer as normal, return result
        return super(Imputer1D, self).transform(X)

# %%==============================================================================
# Empty
# ===============================================================================
class Empty(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """An empty transformer
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        print(self.log)
        return df


# %%==============================================================================
# NumericalToCat
# ===============================================================================
class NumericalToCat(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """Convert numeric indexed column into dtype category with labels
    Convert a column which has a category, presented as an Integer
    Initialize with a dict of ALL mappings for this session, keyed by column name
    (This could be easily refactored to have only the required mapping)
    """
    def __init__(self,label_map_dict):
        self.label_map_dict = label_map_dict

    def fit(self, X, y=None):
        return self

    def transform(self, series):
        assert type(series) == pd.Series
        assert series.name in self.label_map_dict, "{} not in label map!".format(series.Name)
        series = series.astype('category')
        series.cat.rename_categories(self.label_map_dict[series.name], inplace=True)
        print(self.log, series.cat.categories)
        return series

# r  = df_trf.Vaccinated
# r = r.astype('category')
# r.cat.rename_categories(label_maps['Vaccinated'],inplace=True)

# train['FurLength'] = train['FurLength'].astype('category')
# train['FurLength'].cat.rename_categories(map_FurLength,inplace=True)

#%%==============================================================================
# WordCounter
# ===============================================================================
class WordCounter(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """
    """
    def __init__(self, col_name, new_col_name):
        self.col_name = col_name
        self.new_col_name = new_col_name

    def fit(self, X, y=None):
        return self

    def transform(self, df, y=None):
        df[self.new_col_name] = df[self.col_name].apply(lambda x: len(x.split(" ")))
        print(self.log)
        return df


#%% =============================================================================
# TimeProperty
# ===============================================================================
class TimeProperty(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """
    """
    def __init__(self, time_col_name, new_col_name, time_property):
        self.time_col_name = time_col_name
        self.new_col_name = new_col_name
        self.time_property = time_property

    def fit(self, X, y=None):
        return self

    def transform(self, df, y=None):
        original_shape = df.shape
        if self.time_property == 'hour':
            df[self.new_col_name] = df[self.time_col_name].dt.hour
        elif self.time_property == 'month':
            df[self.new_col_name] = df[self.time_col_name].dt.month
        elif self.time_property == 'dayofweek':
            df[self.new_col_name] = df[self.time_col_name].dayofweek
        else:
            raise
        print("Transformer:", type(self).__name__, original_shape, "->", df.shape, vars(self))
        return df


# Debug:
# df = X_train
# time_col_name = 'question_utc'
# new_col_name = 'question_hour'
# time_property = 'hour'
# time_col_name = 'question_utc'
# new_col_name = 'question_month'
# time_property = 'month'
# time_adder = TimeProperty(time_col_name,new_col_name,time_property)
# res=time_adder.transform(df)
#

#%% =============================================================================
# AnswerDelay
# ===============================================================================
class AnswerDelay(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """
    """

    def __init__(self, new_col_name, divisor=1):
        self.new_col_name = new_col_name
        self.divisor = divisor

    def fit(self, X, y=None):
        return self

    def transform(self, df, y=None):
        df[self.new_col_name] = df['answer_utc'] - df['question_utc']
        df[self.new_col_name] = df[self.new_col_name].dt.seconds / self.divisor
        print(self.log)
        return df


# Debug:
# df = X_train
# new_col_name = 'answer_delay_seconds'
# answer_delay_adder = AnswerDelay(new_col_name)
# res=answer_delay_adder.transform(df)
#

#%% =============================================================================
# ValueCounter
# ===============================================================================
class ValueCounter(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """
    """

    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, df, y=None):
        # Count the number of unique entries in a column
        # reset_index() is used to maintain the DataFrame for merging
        selected_df_col = df[self.col_name].value_counts().reset_index()
        # Create a new name for this column
        selected_df_col.columns = [self.col_name, self.col_name + '_counts']
        print(self.log)
        return pd.merge(selected_df_col, df, on=self.col_name)


# ===============================================================================
# ConvertToDatetime
# ===============================================================================
class ConvertToDatetime(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """
    """

    def __init__(self, time_col_name, unit='s'):
        self.time_col_name = time_col_name
        self.unit = unit

    def fit(self, X, y=None):
        return self

    def transform(self, df, y=None):
        df[self.time_col_name] = pd.to_datetime(df[self.time_col_name], unit=self.unit)
        print("Transformer:", type(self).__name__, "converted", self.time_col_name, "to dt")
        return df


# %%=============================================================================
# ConvertDoubleColToDatetime
# ===============================================================================
class ConvertDoubleColToDatetime(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """
    """

    # pd.options.mode.chained_assignment = None  # default='warn'
    def __init__(self, new_col_name, name_col1, name_col2, this_format):
        self.new_col_name = new_col_name
        self.name_col1 = name_col1
        self.name_col2 = name_col2
        self.this_format = this_format

    def fit(self, X, y=None):
        return self

    @timeit
    def transform(self, df, y=None):
        combined_date_string_series = df.loc[:, self.name_col1] + " " + df.loc[:, self.name_col2]
        with ChainedAssignment():
            df.loc[:, self.new_col_name] = pd.to_datetime(combined_date_string_series, format=self.this_format)
        #        pd.options.mode.chained_assignment = 'warn'  # default='warn'

        # print("Transformer:", type(self).__name__, "converted", self.new_col_name, "to dt")
        print(self.log)
        return df

# Debug:
# df = sfpd_head
# new_col_name = 'dt'
# time_adder = ConvertDoubleColToDatetime(new_col_name,name_col1="Date", name_col2="Time",this_format=r'%m/%d/%Y %H:%M')
# res=time_adder.transform(df)
