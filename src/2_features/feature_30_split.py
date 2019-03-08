#%%

# Splitting train and CV
df_tr = df_all[df_all['dataset_type']=='train'].copy()
df_tr.drop('dataset_type', axis=1, inplace=True) # Drop the category
logging.info("Split off train set {}, {:.1%} of the records".format(df_tr.shape,len(df_tr)/len(df_all)))

# SAMPLE_FRACTION = 0.8
if SAMPLE_FRACTION < 1:
    df_tr = df_tr.sample(frac=SAMPLE_FRACTION)
    logging.info("Sampled training set {}, fraction={}".format(df_tr.shape,SAMPLE_FRACTION))

if CV_FRACTION > 0:
    df_tr, df_cv = sklearn.model_selection.train_test_split(df_tr, test_size=CV_FRACTION)
    logging.info("Split off CV set, fraction={}".format(CV_FRACTION))
else:
    df_cv = None

df_te = df_all[df_all['dataset_type']=='test'].copy()
df_te.drop('dataset_type', axis=1, inplace=True)
logging.info("Split off test set {}, {:.1%} of the records".format(df_tr.shape,len(df_te)/len(df_all)))

logging.info("DataFrame summary".format())
logging.info("\tTraining {}".format(df_tr.shape))
if CV_FRACTION > 0:
    logging.info("\tCross Validation {}".format(df_cv.shape))
logging.info("\tTest {}".format(df_te.shape))

#%%
logging.info("Splitting into X_ and y_".format())
#%% Split Train
target_col = 'AdoptionSpeed'
y_tr = df_tr[target_col]
X_tr = df_tr.drop(['AdoptionSpeed'], axis=1)

#%% Split CV
y_cv = df_cv[target_col]
X_cv = df_cv.drop(['AdoptionSpeed'], axis=1)
logging.info("Cross Validation X {}, y {}".format(X_cv.shape, y_cv.shape))

#%% Split Test
# Drop the target (it's NaN anyways)
X_te = df_te.drop(['AdoptionSpeed'], axis=1)

#%%
logging.info("X/y summary".format())


logging.info("\t{:0.1%} Training X {}, y {}".format(len(X_tr)/len(df_all), X_tr.shape, y_tr.shape))
if CV_FRACTION > 0:
    logging.info("\t{:0.1%} Cross Validation X {}, y {}".format(len(X_cv)/len(df_all), X_cv.shape, y_cv.shape))
logging.info("\t{:0.1%} Test X {}".format(len(X_te)/len(df_all), X_te.shape))


#%% DONE HERE - DELETE UNUSED

del_vars =[
    'df_all',
    'df_tr',
    'df_te',
]
cnt = 0
for name in dir():
    if name in del_vars:
        cnt+=1
        del globals()[name]
logging.info(f"Removed {cnt} variables from memory")
del cnt, name, del_vars

