# y_train_data = df_all[df_all['dataset_type']=='train']['AdoptionSpeed'].copy()
#
# mapped_data = y_train_data.cat.codes
# mapped_data.value_counts().sort_index(ascending=True)
# mapped_data.name = 'Mapped'
# # y_train_data.plot.bar()
#
# original_y_train.name = 'Original'
# original_y_train.value_counts().sort_index(ascending=True)
#
# this_df = pd.concat([original_y_train, mapped_data],axis=1)
# # del y_train_data
# dict( enumerate(y_train_data.cat.categories) )
