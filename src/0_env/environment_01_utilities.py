#%% DEBUG TRF
#
# class TransformerLog():
#     """Add a .log attribute for logging
#     """
#     @property
#     def log(self):
#         return "Transformer: {}".format(type(self).__name__)
# class NumericalToCat(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
#     """Convert numeric indexed column into dtype category with labels
#     Convert a column which has a category, presented as an Integer
#     Initialize with a dict of ALL mappings for this session, keyed by column name
#     (This could be easily refactored to have only the required mapping)
#     """
#     def __init__(self,label_map):
#         self.label_map = label_map
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, this_series):
#         assert type(this_series) == pd.Series
#         mapped_labels = list(self.label_map.values())
#         # assert this_series.name in self.label_map_dict, "{} not in label map!".format(this_series.Name)
#         return_series = this_series.copy()
#         return_series = pd.Series(pd.Categorical.from_codes(this_series, mapped_labels))
#         # return_series = return_series.astype('category')
#         # return_series.cat.rename_categories(self.label_map_dict[return_series.name], inplace=True)
#         print(self.log, mapped_labels, return_series.cat.categories, )
#         assert return_series.dtype == 'category'
#         return return_series
#
# # this_series = df_all['Vaccinated'].copy()
# # this_series.value_counts()
# # label_map = label_maps['Vaccinated']
# # mapped_labels = list(label_map.values())
# # my_labels = pd.Index(mapped_labels)
# # pd.Series(pd.Categorical.from_codes(this_series, my_labels))
#
# for col_name in label_maps:
#     df_all[col_name].value_counts().index
#     print(col_name)
#     label_maps[col_name]
#     df_all.replace({col_name: label_maps[col_name]},inplace=True)
#
#
#
# df_all['Vaccinated'] = df_all['Vaccinated'] - 1
#
# pandas.CategoricalIndex.reorder_categories
#
# # To return the original integer mapping!
# ivd = {v: k for k, v in label_maps['State'].items()}
# df_all['State'].astype('object').replace(ivd)