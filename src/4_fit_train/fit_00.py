#%% Fit
if DEPLOYMENT != 'Kaggle':
    import warnings
    warnings.simplefilter("ignore", UserWarning)

    clf_grid.fit(X_tr, y_tr)

    # Print the best parameters found
    print("Best score:", clf_grid.best_score_)
    print("Best parameters:", clf_grid.best_params_)

    clf_grid_BEST = clf_grid.best_estimator_
else:

    # params = {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 20, 'bootstrap': True}
    params = {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 110, 'bootstrap': True}

    clf_grid_BEST = sk.ensemble.RandomForestClassifier(**params)
    clf_grid_BEST.fit(X_tr, y_tr)
#%%


