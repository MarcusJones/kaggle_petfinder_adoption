# %% Predict on X_tr for comparison
y_tr_predicted = clf_grid_BEST.predict(X_tr)

train_kappa = kappa(y_tr, y_tr_predicted)

logging.info("Metric on training set: {:0.3f}".format(train_kappa))

#%% Predict on Test set
# NB we only want the defaulters column!
predicted = clf_grid_BEST.predict(X_te)


#%% Open the submission
# with zipfile.ZipFile(path_data / "test.zip").open("sample_submission.csv") as f:
#     df_submission = pd.read_csv(f, delimiter=',')
df_submission = pd.read_csv(path_data / 'test' / 'sample_submission.csv', delimiter=',')


#%% Collect predicitons
submission = pd.DataFrame({'PetID': df_submission.PetID, 'AdoptionSpeed': [int(i) for i in predicted]})
submission.head()

#%% Create csv
submission.to_csv('submission.csv', index=False)