
#%% Model space

# hyper_param_search = [
#     gamete_design_space.Variable('n_estimators', 'int', [int(x) for x in np.linspace(start=200, stop=2000, num=10)], True),
#     gamete_design_space.Variable('max_features', 'string', ['auto', 'sqrt'], False),
#     gamete_design_space.Variable('max_depth', 'int', [int(x) for x in np.linspace(start=200, stop=2000, num=10)], True),
#     gamete_design_space.Variable('min_samples_split', 'int', [2, 5, 10], True),
#     gamete_design_space.Variable('min_samples_leaf', 'int', [1, 2, 4], True),
#     gamete_design_space.Variable('bootstrap', 'bool', [True, ], False),
# ]

#%%



#%% GridCV
if 0:
    n_estimators_steps = 4
    max_depth_steps = 4
    model_param_search_grid = {
        'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = n_estimators_steps)],
        'max_features' : ['auto', 'sqrt'],
        'max_depth' : [int(x) for x in np.linspace(10, 110, num = max_depth_steps)] + [None],
        'min_samples_split' : [2, 5, 10],
        'min_samples_leaf' : [1, 2, 4],
        'bootstrap' : [True, ],
    #     'bootstrap' : [True, False],
    }

    grid_lengths = [len(key) for key in model_param_search_grid.values()]
    grid_size = reduce(lambda x, y: x*y, grid_lengths)
    logging.info("Grid set, size {}".format(grid_size))


#%%
if 0:
    N_ITER = 400
    N_ITER = 200
    CV_FOLDS = 3
    clf_grid = sk.model_selection.RandomizedSearchCV(estimator=clf, param_distributions=model_param_search_grid,
                                                     n_iter=N_ITER, cv=CV_FOLDS, verbose=50, random_state=42, n_jobs=-1)



    logging.info("Total jobs: {}".format(N_ITER*CV_FOLDS))
    logging.info("Coverage: {:0.1%}".format((N_ITER)/grid_size))

