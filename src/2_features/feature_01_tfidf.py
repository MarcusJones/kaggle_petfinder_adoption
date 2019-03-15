df_all['Description'].fillna('none',inplace=True)
logging.info("Filled 'none' in Description".format())

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
print("Truncated TFIDF to {}".format(X.shape[1]))

#%%
X_df = pd.DataFrame(X, index=df_all.index, columns=['svd_{}'.format(i) for i in range(200)])
print(X_df.shape)
df_all = pd.concat((df_all, X_df), axis =1)
logging.info("df_all {}".format(df_all.shape))
#%%
#OLD:
#%%
if 0:
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier


    PandasSelector(columns='en_US_description',name='Description'),


    classifier = sk.pipeline.Pipeline([
                ('colext', sk.pipeline.TextSelector('Text')),
                ('tfidf', tfv),
                ('svd', TruncatedSVD(algorithm='randomized', n_components=300)),
            ])

    #%%
    v = sk.feature_extraction.text.TfidfVectorizer()
    x = v.fit_transform(res2)

    res2 = pd.DataFrame(df_all_cp['Description'])
    res2.shape
    #%% Build the pipeline
    # this_pipeline = sk.pipeline.make_pipeline(
    #     sk.pipeline.make_pipeline(PandasSelector2(columns='Description', name='Description'), tfv),
    #     # sk.pipeline.make_pipeline(PandasSelector2(columns='Description', name='Description')),
    # )

    sk.pipeline.Pipeline(
        [
         ("asdf", PandasSelector(columns='Description', name='Description')),
         ("asdfasdf", sk.feature_extraction.text.TfidfVectorizer()),
        ]
    )

    logging.info("Created pipeline:")
    for i, step in enumerate(this_pipeline.steps):
        print(i, step[0], step[1].__str__())
    print("pipeline:", [name for name, _ in this_pipeline.steps])



    #%% Another try
    pipeline = sk.pipeline.Pipeline(
            [
             ("selector", ItemSelector(key="text_column")),
             ("vectorizer", TfidfVectorizer()),
             ("debug", InspectPipeline()),
             ("classifier", RandomForestClassifier())
            ]
    )

    #%% Another another try
    # WORKS
    this_pipeline = sk.compose.ColumnTransformer(
        [
            ("tfidf", sk.feature_extraction.text.TfidfVectorizer(), 'Description'),
        ]
    )
    # make_column_transformer(
    #      (StandardScaler(), ['numerical_column']),
    #      (OneHotEncoder(), ['categorical_column'])
    # )
    #%% Fit Transform
    # original_cols = df_all.columns


    df_all_cp_trf = this_pipeline.fit_transform(df_all_cp)
    df_all_cp_trf.shape

    logging.info("Pipeline complete. {} new columns.".format(len(df_all.columns) - len(original_cols)))

    # df_all_cp_trf.head()
    #%%
    from sklearn.decomposition import PCA
    #%%

    # Example!
    if 0:
        # Define a pipeline combining a text feature extractor with a simple
        # classifier
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier()),
        ])

        # uncommenting more parameters will give better exploring power but will
        # increase processing time in a combinatorial way
        parameters = {
            'vect__max_df': (0.5, 0.75, 1.0),
            # 'vect__max_features': (None, 5000, 10000, 50000),
            'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
            # 'tfidf__use_idf': (True, False),
            # 'tfidf__norm': ('l1', 'l2'),
            'clf__max_iter': (5,),
            'clf__alpha': (0.00001, 0.000001),
            'clf__penalty': ('l2', 'elasticnet'),
            # 'clf__max_iter': (10, 50, 80),
        }