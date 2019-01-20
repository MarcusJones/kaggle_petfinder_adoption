#%% Open the submission
with zipfile.ZipFile(path_data / "test.zip").open("sample_submission.csv") as f:
    df_submission = pd.read_csv(f, delimiter=',')

#%% Collect predicitons
prediction = (result_dict_lgb['prediction'])
submission = pd.DataFrame({'PetID': df_submission.PetID, 'AdoptionSpeed': [int(i) for i in prediction]})
submission.head()

#%% Create csv
submission.to_csv('submission.csv', index=False)