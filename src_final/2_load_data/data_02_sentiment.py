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