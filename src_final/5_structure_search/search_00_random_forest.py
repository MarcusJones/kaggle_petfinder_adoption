logging.info(" *** Step 5: (Structure search) ***".format())
# Train 2 seperate models, one for cats, one for dogs!!

# assert y_tr.dtype == np.dtype('int64'), "y_tr must be integer for LGBM!!"


if CONTROL_PARAMS['RUN_TYPE']=='SEARCH':
    #%% Model and params
    params_model = dict()
    # params['num_class'] = len(y_tr.value_counts())
    params_model.update({ })
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

    model_search = ModelSearch(clf, hyper_param_search)

