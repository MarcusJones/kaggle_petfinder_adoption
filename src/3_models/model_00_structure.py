
class ModelStructure:
    def __init__(self, df):
        self.df = df

    def get_sub_df(self,dataset_type):
        sub_df = self.df[self.df['dataset_type'] == dataset_type]
        assert not sub_df._is_view
        return sub_df

    def sample(self, sample_frac):
        """Sample the training set to reduce size

        :return:
        """
        df_tr = self.get_sub_df('train')
        df_tr = df_tr.sample(frac=sample_frac)

        logging.info("Sampled training set {}, fraction={}".format(df_tr.shape, sample_frac))

    def split_cv(self, cv_frac):
            df_tr, df_cv = sklearn.model_selection.train_test_split(df_tr, test_size=CV_FRACTION)
            logging.info("Split off CV set, fraction={}".format(CV_FRACTION))



#%%
this_m = ModelStructure(df_all)
this_m.sample()
this_m.sample(0.8)




#%%


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
logging.info("\t{:0.