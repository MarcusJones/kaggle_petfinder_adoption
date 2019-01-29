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
