X_tr = X_tr
y_tr = y_tr
X_te = X_te

scorer = sk.metrics.make_scorer(kappa, greater_is_better=True, needs_proba=False, needs_threshold=False)

r = sk.model_selection.cross_val_score(clf, X_tr, y_tr, groups=None, scoring=scorer, cv=10, n_jobs=-1, verbose=8, fit_params=None, pre_dispatch='2*n_jobs', error_score='raise-deprecating')
np.mean(r)
r.mean()








