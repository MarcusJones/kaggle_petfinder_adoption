# Train 2 seperate models, one for cats, one for dogs!!

assert y_tr.dtype == np.dtype('int64'), "y_tr must be integer for LGBM!!"
#%%
for col in X_tr()
data_mapper = DataFrameMapper([
    ("district", sk.preprocessing.LabelBinarizer()),
    (["hour"], sk.preprocessing.StandardScaler()),
    (["weekday"], sk.preprocessing.StandardScaler()),
    (["dayofyear"], sk.preprocessing.StandardScaler()),
    (["month"], sk.preprocessing.StandardScaler()),
    (["year"], sk.preprocessing.StandardScaler()),
    (["lon"], sk.preprocessing.StandardScaler()),
    (["lat"], sk.preprocessing.StandardScaler()),
    ("holiday",  sk.preprocessing.LabelEncoder()),
    ("corner", sk.preprocessing.LabelEncoder()),
    ("weekend", sk.preprocessing.LabelEncoder()),
    ("workhour",  sk.preprocessing.LabelEncoder()),
    ("sunlight",  sk.preprocessing.LabelEncoder()),
    ("fri",  sk.preprocessing.LabelEncoder()),
    ("sat",  sk.preprocessing.LabelEncoder()),
    ('Category', sk.preprocessing.LabelEncoder()),
#    ("address", [sk.preprocessing.LabelEncoder(), sk.preprocessing.StandardScaler()]),
#    ("address", sk.preprocessing.LabelEncoder()),
], input_df=True, df_out=True, default=None)


#%% Model and params
params_model = dict()
# params['num_class'] = len(y_tr.value_counts())
params_model.update({

})
clf = sk.ensemble.RandomForestClassifier(**params_model )

#%% GridCV
params_grid = {
    # 'learning_rate': [0.005, 0.05, 0.1, 0.2],
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
