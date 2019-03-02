# Train 2 seperate models, one for cats, one for dogs!!

# assert y_tr.dtype == np.dtype('int64'), "y_tr must be integer for LGBM!!"



#%% Model and params
params_model = dict()
# params['num_class'] = len(y_tr.value_counts())
params_model.update({

})
clf = sk.ensemble.RandomForestClassifier(**params_model )
logging.info("Classifier created: {}".format(clf))

#%% GridCV
n_estimators_steps = 5
max_depth_steps = 5
random_grid = {
    'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = n_estimators_steps)],
    'max_features' : ['auto', 'sqrt'],
    'max_depth' : [int(x) for x in np.linspace(10, 110, num = max_depth_steps)] + [None],
    'min_samples_split' : [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
    'bootstrap' : [True, ],
#     'bootstrap' : [True, False],
}

grid_lengths = [len(key) for key in random_grid.values()]
grid_size = reduce(lambda x, y: x*y, grid_lengths)
logging.info("Grid set, size {}".format(grid_size))

N_ITER = 400
N_ITER = 200
CV_FOLDS = 3
CV_FOLDS = 3
clf_grid = sk.model_selection.RandomizedSearchCV(estimator=clf, param_distributions=random_grid,
                               n_iter=N_ITER, cv=CV_FOLDS, verbose=50, random_state=42, n_jobs=-1)



logging.info("Total jobs: {}".format(N_ITER*CV_FOLDS))
logging.info("Coverage: {:0.1%}".format((N_ITER)/grid_size))

