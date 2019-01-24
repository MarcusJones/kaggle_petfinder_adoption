# Train 2 seperate models, one for cats, one for dogs!!

assert y_tr.dtype == np.dtype('int64'), "y_tr must be integer for LGBM!!"
#%% Model and params
params_model = dict()
# params['num_class'] = len(y_tr.value_counts())
params_model.update({
 'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': -1,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample_for_bin': 200000,
    'objective': 'multiclass',
 'class_weight': None,
    'min_split_gain': 0.0,
    'min_child_weight': 0.001,
    'min_child_samples': 20,
    'subsample': 1.0,
    'subsample_freq': 0,
 'colsample_bytree': 1.0,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'random_state': None,
    'n_jobs': -1, # -1 is for ALL
 'importance_type': 'split',
 'silent': True,
})
clf = lgb.LGBMClassifier(**params_model,

                         )

#%% GridCV
params_grid = {
    'learning_rate': [0.005, 0.05, 0.1, 0.2],
    # 'n_estimators': [40],
    # 'num_leaves': [6,8,12,16],
    # 'boosting_type' : ['gbdt'],
    # 'objective' : ['binary'],
    # 'random_state' : [501], # Updated from 'seed'
    # 'colsample_bytree' : [0.65, 0.66],
    # 'subsample' : [0.7,0.75],
    # 'reg_alpha' : [1,1.2],
    # 'reg_lambda' : [1,1.2,1.4],
    }

clf_grid = sk.model_selection.GridSearchCV(clf, params_grid,
                                       verbose=1,
                                       cv=5,
                                       n_jobs=-1)
#%% Fit
clf_grid.fit(X_tr, y_tr)

# Print the best parameters found
print("Best score:", clf_grid.best_score_)
print("Bast parameters:", clf_grid.best_params_)

clf_grid_BEST = clf_grid.best_estimator_

#%% Do the final fit on the BEST estimator
# start = datetime.datetime.now()
# predicted = clf_grid_BEST.fit(train_X, train_Y)
# logging.info("Elapsed H:m:s: {}".format(datetime.datetime.now()-start))

#%% Predict on Test set
# NB we only want the defaulters column!
predicted = clf_grid_BEST.predict(X_te)

#%% Metric

# kappa(target, train_predictions)
# rmse(target, [r[0] for r in results['train']])
# submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': test_predictions})
# submission.head()


#%%
# n_fold = 5
# folds = sk.model_selection.StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=15)
#
# for fold_n, (train_indices, valid_indices) in enumerate(folds.split(X_tr, y_tr)):
#     logging.info("Fold {:<4} {:0.2f}|{:0.2f}% started {}".format(fold_n,
#                                                        100*len(train_indices)/len(X_tr),
#                                                        100*len(valid_indices)/len(X_tr),
#                                                                   time.ctime()))
#     gc.collect()
#     X_tr_fold, X_val_fold = X_tr.iloc[train_indices], X_tr.iloc[valid_indices]
#     y_tr_fold, y_val_fold = y_tr.iloc[train_indices], y_tr.iloc[valid_indices]
#
#     ds_tr_fold = lgb.Dataset(X_tr_fold, label=y_tr_fold)
#     ds_val_data = lgb.Dataset(X_val_fold, label=y_val_fold)
#
#
#     pprint(model.get_params())
#     logging.info("Model instantiated".format())
#
#
#     # model = lgb.train(params,
#     #                   ds_tr_fold,
#     #                   num_boost_round=2000,
#     #                   valid_sets=[ds_tr_fold, ds_val_data],
#     #                   verbose_eval=100,
#     #                   early_stopping_rounds=200)
#
#         grid = sk.model_selection.GridSearchCV(mdl, gridParams,
#                             verbose=0,
#                             cv=4,
#                             n_jobs=2)
