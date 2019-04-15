# %%
# Ensure the target is unchanged
# assert all(y_tr.sort_index() == original_y_train.sort_index())
# Ensure the target is unchanged (unshuffled!)
# assert all(y_tr == original_y_train)

# %% Predict on X_tr for comparison
y_tr_predicted = clf.predict(X_tr)

# original_y_train.value_counts()
# y_tr.cat.codes.value_counts()
# y_tr_predicted.value_counts()
# y_tr.value_counts()

train_kappa = kappa(y_tr, y_tr_predicted)

logging.info("Metric on training set: {:0.3f}".format(train_kappa))
# these_labels = list(label_maps['AdoptionSpeed'].values())
sk.metrics.confusion_matrix(y_tr, y_tr_predicted)


#%% Predict on X_cv for cross validation
# if CV_FRACTION > 0:
#     y_cv_predicted = clf.predict(X_cv)
#     train_kappa_cv = kappa(y_cv, y_cv_predicted)
#     logging.info("Metric on Cross Validation set: {:0.3f}".format(train_kappa_cv))
#     sk.metrics.confusion_matrix(y_cv, y_cv_predicted)

#%% Predict on Test set
# NB we only want the defaulters column!
logging.info("Predicting on X_te".format())
predicted = clf.predict(X_te)

# raise "Lost the sorting of y!"


logging.info("--- Fit summary ---".format())
if 'CV SCORE' in CONTROL_PARAMS:
    logging.info("CV Score: {:0.5f}".format(RESULTS['CV Score']))
    logging.info("CV standard deviation: {:0.5f}".format(RESULTS['CV standard deviation']))


#%% Open the submission
# with zipfile.ZipFile(path_data / "test.zip").open("sample_submission.csv") as f:
#     df_submission = pd.read_csv(f, delimiter=',')

logging.info("Creating submission".format())
df_submission_template = pd.read_csv(path_data / 'test' / 'sample_submission.csv', delimiter=',')
df_submission = pd.DataFrame({'PetID': df_submission_template.PetID, 'AdoptionSpeed': [int(i) for i in predicted]})

#%% Collect predicitons
df_submission.head()

#%% Create csv
logging.info("Saving submission to csv.".format())
df_submission.to_csv('submission.csv', index=False)

CONTROL_PARAMS['KERNEL_END_TIME'] =datetime.datetime.now()

CONTROL_PARAMS['TOTAL_ELAPSED_TIME'] = CONTROL_PARAMS['START_TIME'] - CONTROL_PARAMS['KERNEL_END_TIME']

logging.info("All done, {} elapsed".format(CONTROL_PARAMS['TOTAL_ELAPSED_TIME']/60))