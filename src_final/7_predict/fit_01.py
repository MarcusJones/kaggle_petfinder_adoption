#%%
RESULTS = dict()
#%%

if 'CV SCORE' in CONTROL_PARAMS:
    logging.info("Calculating CV score for the classifier".format())

    scorer = sk.metrics.make_scorer(kappa, greater_is_better=True, needs_proba=False, needs_threshold=False)

    start = time.time()

    r = sk.model_selection.cross_val_score(clf, X_tr, y_tr,
                                           groups=None,
                                           scoring=scorer,
                                           cv=CONTROL_PARAMS['CV FOLDS'],
                                           n_jobs=-1,
                                           verbose=8,
                                           fit_params=None,
                                           pre_dispatch='2*n_jobs',
                                           error_score='raise-deprecating')
    logging.info("Ran CV selection, {:0.1f} minutes elapsed".format((time.time() - start)/60))
    logging.info("Mean of CV folds:{:0.3f}".format(np.mean(r)))
    logging.info("STD of CV folds:{:0.3f}".format(np.std(r)))
    logging.info("Mean {:0.5f} +/- {:0.1%}".format(np.mean(r), np.std(r)/np.mean(r)))

    RESULTS['CV Score'] = np.mean(r)
    RESULTS['CV standard deviation'] = np.std(r)


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

    clf = clf_grid.best_estimator_

# Simple
elif CONTROL_PARAMS['RUN_TYPE'] == 'SIMPLE':

    logging.info("Simple run for {}".format(CONTROL_PARAMS['DEPLOYMENT']))
    # params = {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 20, 'bootstrap': True}
    # params = {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 110, 'bootstrap': True}
    logging.info("Running fit with parameters: {}".format(params))
    # clf_grid_BEST = sk.ensemble.RandomForestClassifier(**params)
    start = time.time()
    clf.fit(X_tr, y_tr)
    logging.info("Fit finished, {:0.1f}m elapsed".format((time.time() - start)/60))

else:
    raise

#%%
assert 0 < len( [k for k,v in inspect.getmembers(clf) if k.endswith('_') and not k.startswith('__')] ), "Classifier is not fitted"
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