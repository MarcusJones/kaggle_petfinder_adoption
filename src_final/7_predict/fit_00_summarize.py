CONTROL_PARAMS['START_FIT_TIME'] =datetime.datetime.now()

logging.info("--- Control parameter summary ---".format())
for k in CONTROL_PARAMS:
    logging.info("{}={}".format(k, CONTROL_PARAMS[k]))

logging.info("--- Split data summary ---".format())
#%% Sample
df_all.columns
# ds.sample_train(0.8)

X_tr, y_tr, X_te, y_te = ds.split_train_test()

logging.info("X_tr {}".format(X_tr.shape))
logging.info("y_tr {}".format(y_tr.shape))
logging.info("X_te {}".format(X_te.shape))
logging.info("y_te {}".format(y_te.shape))
