# Train 2 seperate models, one for cats, one for dogs!!

# assert y_tr.dtype == np.dtype('int64'), "y_tr must be integer for LGBM!!"



#%% Model and params
params_model = dict()
# params['num_class'] = len(y_tr.value_counts())
params_model.update({

})
clf = sk.ensemble.RandomForestClassifier(**params_model )

#%% GridCV
random_grid = {
    'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
    'max_features' : ['auto', 'sqrt'],
    'max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)] + [None],
    'min_samples_split' : [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
    'bootstrap' : [True, False],
}
clf_grid = sk.model_selection.RandomizedSearchCV(estimator=clf, param_distributions=random_grid,
                               n_iter=50, cv=3, verbose=1, random_state=42, n_jobs=-1)

# clf_grid = sk.model_selection.GridSearchCV(clf, params_grid,
#                                        verbose=1,
#                                        cv=5,
#                                        n_jobs=-1)
