params = {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 110,
          'bootstrap': True}
logging.info("Running fit with parameters: {}".format(params))
clf_grid_BEST = sk.ensemble.RandomForestClassifier(**params)