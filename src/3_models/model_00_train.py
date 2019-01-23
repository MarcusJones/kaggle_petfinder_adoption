# Train 2 seperate models, one for cats, one for dogs!!

assert y_tr.dtype == np.dtype('int64'), "y_tr must be integer for LGBM!!"

params_model = dict()
# params['num_class'] = len(y_tr.value_counts())
params_model.update({
 'boosting_type': 'gbdt',
 'class_weight': None,
 'colsample_bytree': 1.0,
 'importance_type': 'split',
 'learning_rate': 0.1,
 'max_depth': -1,
 'min_child_samples': 20,
 'min_child_weight': 0.001,
 'min_split_gain': 0.0,
 'n_estimators': 100,
 'n_jobs': -1,
 'num_leaves': 31,
 'objective': None,
 'random_state': None,
 'reg_alpha': 0.0,
 'reg_lambda': 0.0,
 'silent': True,
 'subsample': 1.0,
 'subsample_for_bin': 200000,
 'subsample_freq': 0})

params_grid = {
    'learning_rate': [0.005],
    'n_estimators': [40],
    'num_leaves': [6,8,12,16],
    'boosting_type' : ['gbdt'],
    'objective' : ['binary'],
    'random_state' : [501], # Updated from 'seed'
    'colsample_bytree' : [0.65, 0.66],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [1,1.2],
    'reg_lambda' : [1,1.2,1.4],
    }


#%%
n_fold = 5
folds = sk.model_selection.StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=15)

for fold_n, (train_indices, valid_indices) in enumerate(folds.split(X_tr, y_tr)):
    logging.debug("Fold {:<4} {:0.2f}|{:0.2f}% started {}".format(fold_n,
                                                       100*len(train_indices)/len(X_tr),
                                                       100*len(valid_indices)/len(X_tr),
                                                                  time.ctime()))
    gc.collect()
    X_tr_fold, X_val_fold = X_tr.iloc[train_indices], X_tr.iloc[valid_indices]
    y_tr_fold, y_val_fold = y_tr.iloc[train_indices], y_tr.iloc[valid_indices]

    ds_tr_fold = lgb.Dataset(X_tr_fold, label=y_tr_fold)
    ds_val_data = lgb.Dataset(X_val_fold, label=y_val_fold)

    model = lgb.LGBMClassifier(params_model)
    pprint(model.get_params())
    logging.debug("Model instantiated".format())


    # model = lgb.train(params,
    #                   ds_tr_fold,
    #                   num_boost_round=2000,
    #                   valid_sets=[ds_tr_fold, ds_val_data],
    #                   verbose_eval=100,
    #                   early_stopping_rounds=200)

        grid = sk.model_selection.GridSearchCV(mdl, gridParams,
                            verbose=0,
                            cv=4,
                            n_jobs=2)
