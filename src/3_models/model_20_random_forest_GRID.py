# Train 2 seperate models, one for cats, one for dogs!!

# assert y_tr.dtype == np.dtype('int64'), "y_tr must be integer for LGBM!!"

class ModelSearch():
    def __init__(self, classifier, search_grid):
        self.classifier = classifier
        self.search_grid = search_grid


#%% Model and params
params_model = dict()
    # params['num_class'] = len(y_tr.value_counts())
    params_model.update({
    })
clf = sk.ensemble.RandomForestClassifier(**params_model )
logging.info("Classifier created: {}".format(clf))
#%%
hyper_param_search = [
    {'name':'n_estimators', 'vtype':'int', 'variable_tuple':[int(x) for x in np.linspace(start=200, stop=2000, num=10)], 'ordered':True},
    {'name':'max_features', 'vtype':'string', 'variable_tuple':['auto', 'sqrt'], 'ordered':False},
    {'name':'max_depth', 'vtype':'int', 'variable_tuple':[int(x) for x in np.linspace(start=200, stop=2000, num=10)], 'ordered':True},
    {'name':'min_samples_split', 'vtype':'int', 'variable_tuple':[2, 5, 10], 'ordered':True},
    {'name':'min_samples_leaf', 'vtype':'int', 'variable_tuple':[1, 2, 4], 'ordered':True},
    {'name':'bootstrap', 'vtype':'bool', 'variable_tuple':[True, ], 'ordered':False},
]


#%% Model space

hyper_param_search = [
    gamete_design_space.Variable('n_estimators', 'int', [int(x) for x in np.linspace(start=200, stop=2000, num=10)], True),
    gamete_design_space.Variable('max_features', 'string', ['auto', 'sqrt'], False),
    gamete_design_space.Variable('max_depth', 'int', [int(x) for x in np.linspace(start=200, stop=2000, num=10)], True),
    gamete_design_space.Variable('min_samples_split', 'int', [2, 5, 10], True),
    gamete_design_space.Variable('min_samples_leaf', 'int', [1, 2, 4], True),
    gamete_design_space.Variable('bootstrap', 'bool', [True, ], False),
]

#%%
model_search = ModelSearch(clf, hyper_param_search)


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

