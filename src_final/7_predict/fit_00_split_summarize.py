logging.info(" *** Step 7: Fit and predict ***".format())

CONTROL_PARAMS['START_FIT_TIME'] =datetime.datetime.now()

logging.info("--- Control parameter summary ---".format())
for k in CONTROL_PARAMS:
    logging.info("{}={}".format(k, CONTROL_PARAMS[k]))

#%% Sample and split
logging.info("--- Split data summary ---".format())
df_all.columns

if CONTROL_PARAMS['SAMPLE_FRACTION'] < 1:
    ds.sample_train(CONTROL_PARAMS['SAMPLE_FRACTION'])

X_tr, y_tr, X_te, y_te = ds.split_train_test()

logging.info("X_tr {}".format(X_tr.shape))
logging.info("y_tr {}".format(y_tr.shape))
logging.info("X_te {}".format(X_te.shape))
logging.info("y_te {}".format(y_te.shape))

logging.info("--- Model summary ---".format())
logging.info("{}".format(clf))
