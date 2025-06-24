import pandas as pd
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

df = pd.read_csv("clickbait_data.csv", encoding='utf-8')

#punctuation analysis
def count_punctuation(text, punct):
    return text.count(punct)

punctuations = ['!', '?', '.', '...']
for p in punctuations:
    df[f'count_{p}'] = df['Headline'].astype(str).apply(lambda x: count_punctuation(x, p))

punct_summary = df.groupby('label')[[f'count_{p}' for p in punctuations]].mean().T
punct_summary.columns = ['Non-clickbait', 'Clickbait']

punct_summary.plot(kind='bar', figsize=(8, 5), title='Average Punctuation Use per Headline')
plt.ylabel('Average Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#n-gram analysis
def top_ngrams(corpus, ngram_range=(2, 2), n=10):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    X = vec.fit_transform(corpus)
    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    return sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]

clickbait_ngrams = top_ngrams(df[df['label'] == 1]['Headline'], ngram_range=(2, 2))
non_clickbait_ngrams = top_ngrams(df[df['label'] == 0]['Headline'], ngram_range=(2, 2))

print("\n Top Bigrams in Clickbait Headlines:")
for ng, count in clickbait_ngrams:
    print(f"{ng}: {count}")

print("\n Top Bigrams in Non-Clickbait Headlines:")
for ng, count in non_clickbait_ngrams:
    print(f"{ng}: {count}")

#sentiment analysis
def get_sentiment(text):
    blob = TextBlob(str(text))
    return blob.sentiment.polarity, blob.sentiment.subjectivity

df[['polarity', 'subjectivity']] = df['Headline'].apply(lambda x: pd.Series(get_sentiment(x)))

sentiment_summary = df.groupby('label')[['polarity', 'subjectivity']].mean()
sentiment_summary.index = ['Non-clickbait', 'Clickbait']
print("\n Average Sentiment Scores:")
print(sentiment_summary)

sentiment_summary.plot(kind='bar', figsize=(8, 5), title='Sentiment Analysis: Clickbait vs Non-Clickbait')
plt.ylabel('Average Score')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
