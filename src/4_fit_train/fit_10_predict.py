# %%
# Ensure the target is unchanged
assert all(y_tr.sort_index() == original_y_train.sort_index())
# Ensure the target is unchanged (unshuffled!)
assert all(y_tr == original_y_train)

# %% Predict on X_tr for comparison
y_tr_predicted = clf_grid_BEST.predict(X_tr)

# original_y_train.value_counts()
# y_tr.cat.codes.value_counts()
# y_tr_predicted.value_counts()
# y_tr.value_counts()

train_kappa = kappa(y_tr, y_tr_predicted)

logging.info("Metric on training set: {:0.3f}".format(train_kappa))
# these_labels = list(label_maps['AdoptionSpeed'].values())
sk.metrics.confusion_matrix(y_tr, y_tr_predicted)

#%% Predict on Test set
# NB we only want the defaulters column!
predicted = clf_grid_BEST.predict(X_te)

# raise "Lost the sorting of y!"
#%% Open the submission
# with zipfile.ZipFile(path_data / "test.zip").open("sample_submission.csv") as f:
#     df_submission = pd.read_csv(f, delimiter=',')
df_submission_template = pd.read_csv(path_data / 'test' / 'sample_submission.csv', delimiter=',')
df_submission = pd.DataFrame({'PetID': df_submission_template.PetID, 'AdoptionSpeed': [int(i) for i in predicted]})

#%% Collect predicitons
df_submission.head()

#%% Create csv
df_submission.to_csv('submission.csv', index=False)