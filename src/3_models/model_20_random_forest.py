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
    'n_estimators': [int(x) for x in np.linspace(start = 800, stop = 1500, num = 8)],
    'max_features' : ['auto', 'sqrt'],
    'max_depth' : [int(x) for x in np.linspace(10, 40, num = 8)] + [None],
    'min_samples_split' : [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
    'bootstrap' : [True, ],
    # 'bootstrap' : [True, False],
}

grid_lengths = [len(key) for key in random_grid.values()]
grid_size = reduce(lambda x, y: x*y, grid_lengths)
logging.info("Grid size {}".format(grid_size))
# Best parameters: {'n_estimators': 1000, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 20, 'bootstrap': True}

clf_grid = sk.model_selection.RandomizedSearchCV(estimator=clf, param_distributions=random_grid,
                               n_iter=50, cv=3, verbose=1, random_state=42, n_jobs=-1)

# clf_grid = sk.model_selection.GridSearchCV(clf, params_grid,
#                                        verbose=1,
#                                        cv=5,
#                                        n_jobs=-1)
