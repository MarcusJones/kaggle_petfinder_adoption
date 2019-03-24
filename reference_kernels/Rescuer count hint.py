# Brett, I also shared your feeling that there should be some kind of scaling of the rescuer counts
# so that the values for train and test have more similar meanings.
#
# I finally decided on the concept of "RescuerActivity" as a measure of how active a rescuer is. Very low values of rescued pets (1,2) are the lowest activity and those should have similar meaning between test and training. But a larger value like 60 rescues in the training set would more likely correspond to a scaled number around 16 in the test set.
#
# So, in the code below I scale the training counts (above a low value) so that the training numbers will be on a similar scale as the test counts. Instead of scaling by the total number of samples, I scale by the number of samples above the low value. Hopefully this scaling roughly equates typically active rescuers between the data sets. There's also an 0.5 thrown in and then I take the log to have a smaller range of values.
#
# Including this feature helped my simple model go from 0.398 to 0.416, and it is the third most important feature. That improvement is most likely just due to including a RescuerID-count-related feature rather than the specific scaling method I used; but I didn't LB-try any other versions of a RescuerID count since this scheme felt "right" ;-)

# RescuerID - "to use or not to use?"
# See the discussion at:
# https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/78511#486890
# A valid reason to include features based on it is that it gives information on the rescuer.
# Create RescuerActivity columns based on the code in:
# https://www.kaggle.com/wrosinski/baselinemodeling
#
# Train
rescuer_count = df_train.groupby(['RescuerID'])['PetID'].count().reset_index()
rescuer_count.columns = ['RescuerID', 'RescuerActivity']
df_train = df_train.merge(rescuer_count, how='left', on='RescuerID')
# Test
rescuer_count = df_test.groupby(['RescuerID'])['PetID'].count().reset_index()
rescuer_count.columns = ['RescuerID', 'RescuerActivity']
df_test = df_test.merge(rescuer_count, how='left', on='RescuerID')
#
# Do a log scaling on these, but first
# Scale-down the train values by the test/train ratio
# of the number of values above a low number (e.g., 2)
col = 'RescuerActivity'
##test2train = len(df_test)/len(df_train)
n_low = 2
test2train = len(df_test.loc[df_test[col] > n_low, col])/ \
                    len(df_train.loc[df_train[col] > n_low, col])
df_train.loc[df_train[col] > n_low, col] = n_low + 0.5 + \
                    test2train * (df_train.loc[df_train[col] > n_low, col] - n_low)
df_train[col] = np.log(df_train[col].astype(int))
# Keep the test values as they are and do log()
df_test[col] = np.log(df_test[col])