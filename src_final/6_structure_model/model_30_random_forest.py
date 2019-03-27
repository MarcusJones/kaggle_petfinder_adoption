logging.info(" *** Step 6: Specify model ***".format())
params = {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 110,
          'bootstrap': True}
clf = sk.ensemble.RandomForestClassifier(**params)

logging.info("Classifier with parameters: {}".format(params))

