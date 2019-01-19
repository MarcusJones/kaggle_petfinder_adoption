import sklearn as sk
import sklearn.preprocessing
import pandas as pd
import time
import numpy as np
from sklearn_pandas import DataFrameMapper
class TransformerLog():
    """Add a .log attribute for logging
    """
    @property
    def log(self):
        return "Transformer: {}".format(type(self).__name__)
# ===============================================================================
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