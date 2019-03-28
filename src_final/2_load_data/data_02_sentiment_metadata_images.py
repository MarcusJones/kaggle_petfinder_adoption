# %% {"_uuid": "c3399c9ff73a9dd37cecb657cc26e90b934f67df"}
train_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_images/*.jpg'))
train_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_metadata/*.json'))
train_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_sentiment/*.json'))

print(f'num of train images files: {len(train_image_files)}')
print(f'num of train metadata files: {len(train_metadata_files)}')
print(f'num of train sentiment files: {len(train_sentiment_files)}')


test_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_images/*.jpg'))
test_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_metadata/*.json'))
test_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_sentiment/*.json'))

print(f'num of test images files: {len(test_image_files)}')
print(f'num of test metadata files: {len(test_metadata_files)}')
print(f'num of test sentiment files: {len(test_sentiment_files)}')

# %% [markdown] {"_uuid": "5a74b43503752f801202ba62495d47836f8704c9"}
# ### Train

# %% {"_uuid": "bc13e1b9227cc808bcba7204e7fd499c597b1796"}
# Images:
train_df_ids = train[['PetID']]
print(train_df_ids.shape)

# Metadata:
train_df_ids = train[['PetID']]
train_df_metadata = pd.DataFrame(train_metadata_files)
train_df_metadata.columns = ['metadata_filename']
train_metadata_pets = train_df_metadata['metadata_filename'].apply(lambda x: x.split(split_char)[-1].split('-')[0])
train_df_metadata = train_df_metadata.assign(PetID=train_metadata_pets)
print(len(train_metadata_pets.unique()))

pets_with_metadatas = len(np.intersect1d(train_metadata_pets.unique(), train_df_ids['PetID'].unique()))
print(f'fraction of pets with metadata: {pets_with_metadatas / train_df_ids.shape[0]:.3f}')

# Sentiment:
train_df_ids = train[['PetID']]
train_df_sentiment = pd.DataFrame(train_sentiment_files)
train_df_sentiment.columns = ['sentiment_filename']
train_sentiment_pets = train_df_sentiment['sentiment_filename'].apply(lambda x: x.split(split_char)[-1].split('.')[0])
train_df_sentiment = train_df_sentiment.assign(PetID=train_sentiment_pets)
print(len(train_sentiment_pets.unique()))

pets_with_sentiments = len(np.intersect1d(train_sentiment_pets.unique(), train_df_ids['PetID'].unique()))
print(f'fraction of pets with sentiment: {pets_with_sentiments / train_df_ids.shape[0]:.3f}')

# %% [markdown] {"_uuid": "828ef0c92408c1b67a0f3c80efe608792a43837d"}
# ### Test

# %% {"_uuid": "514c3f5a3d8bf6b396425d1693f5491e874f4cc0"}
# Images:
test_df_ids = test[['PetID']]
print(test_df_ids.shape)

# Metadata:
test_df_metadata = pd.DataFrame(test_metadata_files)
test_df_metadata.columns = ['metadata_filename']
test_metadata_pets = test_df_metadata['metadata_filename'].apply(lambda x: x.split(split_char)[-1].split('-')[0])
test_df_metadata = test_df_metadata.assign(PetID=test_metadata_pets)
print(len(test_metadata_pets.unique()))

pets_with_metadatas = len(np.intersect1d(test_metadata_pets.unique(), test_df_ids['PetID'].unique()))
print(f'fraction of pets with metadata: {pets_with_metadatas / test_df_ids.shape[0]:.3f}')

# Sentiment:
test_df_sentiment = pd.DataFrame(test_sentiment_files)
test_df_sentiment.columns = ['sentiment_filename']
test_sentiment_pets = test_df_sentiment['sentiment_filename'].apply(lambda x: x.split(split_char)[-1].split('.')[0])
test_df_sentiment = test_df_sentiment.assign(PetID=test_sentiment_pets)
print(len(test_sentiment_pets.unique()))

pets_with_sentiments = len(np.intersect1d(test_sentiment_pets.unique(), test_df_ids['PetID'].unique()))
print(f'fraction of pets with sentiment: {pets_with_sentiments / test_df_ids.shape[0]:.3f}')


#%%












#%%

file_sentiment = file['documentSentiment']
file_entities = [x['name'] for x in file['entities']]
file_entities = self.sentence_sep.join(file_entities)

file_sentences_sentiment = [x['sentiment'] for x in file['sentences']]

file_sentences_sentiment = pd.DataFrame.from_dict(
    file_sentences_sentiment, orient='columns')
file_sentences_sentiment_df = pd.DataFrame(
    {
        'magnitude_sum': file_sentences_sentiment['magnitude'].sum(axis=0),
        'score_sum': file_sentences_sentiment['score'].sum(axis=0),
        'magnitude_mean': file_sentences_sentiment['magnitude'].mean(axis=0),
        'score_mean': file_sentences_sentiment['score'].mean(axis=0),
        'magnitude_var': file_sentences_sentiment['magnitude'].var(axis=0),
        'score_var': file_sentences_sentiment['score'].var(axis=0),
    }, index=[0]
)

df_sentiment = pd.DataFrame.from_dict(file_sentiment, orient='index').T
df_sentiment = pd.concat([df_sentiment, file_sentences_sentiment_df], axis=1)

df_sentiment['entities'] = file_entities
df_sentiment = df_sentiment.add_prefix('sentiment_')