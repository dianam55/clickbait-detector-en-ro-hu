import pandas as pd
from collections import Counter
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from transformers import pipeline
import torch

nlp = spacy.load("ro_core_news_sm")

df = pd.read_csv("ro_combined_file.csv", encoding='utf-8')
df['Headline'] = df['Headline'].astype(str)

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
ro_stop_words = nlp.Defaults.stop_words

def top_ngrams(corpus, ngram_range=(2, 2), n=10):
    def tokenize(text):
        doc = nlp(text)
        return [token.text.lower() for token in doc if not token.is_punct and not token.is_stop]
      
    processed_corpus = [' '.join(tokenize(text)) for text in corpus]
  
    vec = CountVectorizer(ngram_range=ngram_range, stop_words=list(ro_stop_words))
    X = vec.fit_transform(processed_corpus)
    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    return sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]

clickbait_ngrams = top_ngrams(df[df['label'] == 1]['Headline'], ngram_range=(2, 2))
non_clickbait_ngrams = top_ngrams(df[df['label'] == 0]['Headline'], ngram_range=(2, 2))

print("\nCele mai frecvente bigrame în titluri clickbait:")
for ng, count in clickbait_ngrams:
    print(f"{ng}: {count}")

print("\nCele mai frecvente bigrame în titluri non-clickbait:")
for ng, count in non_clickbait_ngrams:
    print(f"{ng}: {count}")

#sentiment analysis
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="dumitrescustefan/bert-base-romanian-uncased-v1",
    tokenizer="dumitrescustefan/bert-base-romanian-uncased-v1",
    device=0 if torch.cuda.is_available() else -1  
)

def get_sentiment(text):
    try:
        result = sentiment_analyzer(text, truncation=True, max_length=512)
        label = result[0]['label']
        score = result[0]['score']
        polarity = score if label == 'POSITIVE' else -score
        subjectivity = 1 - score if score > 0.5 else score
        return polarity, subjectivity
    except Exception:
        return 0.0, 0.0 

df[['polarity', 'subjectivity']] = df['Headline'].apply(lambda x: pd.Series(get_sentiment(x)))

sentiment_summary = df.groupby('label')[['polarity', 'subjectivity']].mean()
sentiment_summary.index = ['Non-clickbait', 'Clickbait']
print("\nAverage Sentiment Scores:")
print(sentiment_summary)

sentiment_summary.plot(kind='bar', figsize=(8, 5), title='Sentiment Analysis: Clickbait vs Non-Clickbait')
plt.ylabel('Average Score')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
