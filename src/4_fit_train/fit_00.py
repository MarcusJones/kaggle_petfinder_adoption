#%% Fit
clf_grid.fit(X_tr, y_tr)

# Print the best parameters found
print("Best score:", clf_grid.best_score_)
print("Best parameters:", clf_grid.best_params_)

clf_grid_BEST = clf_grid.best_estimator_