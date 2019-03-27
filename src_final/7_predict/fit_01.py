

#%% Search
if CONTROL_PARAMS['RUN_TYPE'] == "SEARCH":
    logging.info("Grid Search for {}".format(DEPLOYMENT))
    # For local training, run a gridsearch
    import warnings
    warnings.simplefilter("ignore", UserWarning)

    # Fit the grid!
    logging.info("Running grid fit".format())
    clf_grid.fit(X_tr, y_tr)

    # Print the best parameters found
    print("Best score:", clf_grid.best_score_)
    print("Best parameters:", clf_grid.best_params_)
    print("", clf_grid.cv_results_)

    clf_grid_BEST = clf_grid.best_estimator_

# Simple
elif CONTROL_PARAMS['RUN_TYPE'] == 'SIMPLE':

    logging.info("Simple run for {}".format(CONTROL_PARAMS['DEPLOYMENT']))
    # params = {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 20, 'bootstrap': True}
    params = {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 110, 'bootstrap': True}
    logging.info("Running fit with parameters: {}".format(params))
    clf_grid_BEST = sk.ensemble.RandomForestClassifier(**params)
    clf_grid_BEST.fit(X_tr, y_tr)

elif CONTROL_PARAMS['RUN_TYPE'] == 'KFOLDS':
    logging.info("Simple run for {}".format(CONTROL_PARAMS['DEPLOYMENT']))
    # params = {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 20, 'bootstrap': True}
    params = {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 110, 'bootstrap': True}
    logging.info("Running fit with parameters: {}".format(params))
    clf_grid_BEST = sk.ensemble.RandomForestClassifier(**params)
    clf_grid_BEST.fit(X_tr, y_tr)

else:
    raise

#%%
logging.info("Fit finished. ".format())

#%%
if 0:
    features = X_tr.columns
    importances = clf_grid_BEST.feature_importances_
    indices = np.argsort(importances)
    if DEPLOYMENT != 'Kaggle' and 0:
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()