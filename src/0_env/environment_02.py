if 0:
    #%%
    @depcretiated
    def cross_validate(_X_tr, _y_tr, model, metric_function, folds=10, repeats=5):
        """
        Function to do the cross validation - using stacked Out of Bag method instead of averaging across folds.
        model = algorithm to validate. Must be scikit learn or scikit-learn like API (Example xgboost XGBRegressor)
        x = training data, numpy array
        y = training labels, numpy array
        folds = K, the number of folds to divide the data into
        repeats = Number of times to repeat validation process for more confidence
        :param _X_tr:
        :param _y_tr:
        :param model: SKLearn API Model
        :param metric_function: takes y, y_pred
        :param folds:
        :param repeats:
        :return:
        """
        y_cv_fold_pred = np.zeros((len(_y_tr), repeats))
        score = np.zeros(repeats)
        _X_tr = np.array(_X_tr)
        for r in range(repeats):
            i=0
            logging.info("Cross Validating - Run {} of {}".format(str(r + 1), str(repeats)))
            _X_tr, _y_tr = sk.utils.shuffle(_X_tr, _y_tr, random_state=r)    #shuffle data before each repeat
            kf = sk.model_selection.KFold(n_splits=folds, random_state=i+1000)         #random split, different each time
            for train_ind, cv_ind in kf.split(_X_tr):
                print('Fold', i+1, 'out of', folds)
                _X_tr_fold, _y_tr_fold = _X_tr[train_ind, :], _y_tr[train_ind]
                _X_cv_fold, _y_cv_fold = _X_tr[cv_ind, :], _y_tr[cv_ind]
                model.fit(_X_tr_fold, _y_tr_fold)
                y_cv_fold_pred[cv_ind, r] = model.predict(_X_cv_fold)
                i+=1
            score[r] = metric_function(y_cv_fold_pred[:,r], _y_tr)

        logging.info('\nCV complete, overall score: {}'.format(str(score)))
        logging.info('Mean: {}'.format(str(np.mean(score))))
        logging.info('Deviation: {}'.format(str(np.std(score))))

    #%%
    """
    CV complete, overall score: [0.31460838 0.31986689 0.31859581 0.32246347 0.32236281]
    2019-03-09 22:01:39 : Mean: 0.31957947269358394
    2019-03-09 22:01:39 : Deviation: 0.002892279348388643
    """
    cross_validate(X_tr, y_tr, clf, kappa)


