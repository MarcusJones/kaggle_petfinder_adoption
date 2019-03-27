CONTROL_PARAMS['START_FIT_TIME'] =datetime.datetime.now()

logging.info("Control parameter summary".format())
for k in CONTROL_PARAMS:
    logging.info("{}={}".format(k, CONTROL_PARAMS[k]))


X_tr, y_tr, X_te, y_te = ds.split_train_test()