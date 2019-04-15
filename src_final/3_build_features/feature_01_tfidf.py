df_all['Description'].fillna('none',inplace=True)
logging.info("Filled na with 'none' in Description".format())

#%%
df_all_cp = df_all.copy()

#%%
# Dimensionality reduction using truncated SVD (aka LSA).
#This transformer performs linear dimensionality reduction by means of truncated singular value decomposition (SVD). Contrary to PCA, this estimator does not center the data before computing the singular value decomposition. This means it can work with scipy.sparse matrices efficiently.
#In particular, truncated SVD works on term count/tf-idf matrices as returned by the vectorizers in sklearn.feature_extraction.text. In that context, it is known as latent semantic analysis (LSA).
# min_df=2
# max_features=None
tfv = sk.feature_extraction.text.TfidfVectorizer(min_df=2, max_features=None,
                                                 strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                                                 ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                                                 )

#%%
# Fit the TFIDF
logging.info("Fit Transforming the TFIDF".format())
X = tfv.fit_transform(df_all_cp['Description'])
logging.info("TFIDF shape: {}".format(X.shape))

#%%
n_components = 200
svd = sk.decomposition.TruncatedSVD(n_components=n_components, random_state=42)
logging.info("Fitting SVD on X".format())
svd.fit(X)
logging.info("SUM Percentage of variance explained by each of the selected components {}".format(svd.explained_variance_ratio_.sum()))

#%%
X = svd.transform(X)
logging.info("Truncated TFIDF to {}".format(X.shape[1]))

#%%
X_df = pd.DataFrame(X, index=df_all.index, columns=['svd_{}'.format(i) for i in range(200)])
df_all = pd.concat((df_all, X_df), axis =1)
logging.info("df_all {}".format(df_all.shape))
