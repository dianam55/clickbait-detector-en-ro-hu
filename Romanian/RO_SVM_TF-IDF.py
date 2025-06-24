#!pip install spacy
#!python -m spacy download ro_core_news_lg

import pandas as pd
import spacy
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files

nlp = spacy.load("ro_core_news_lg")

uploaded = files.upload()
df_Clickbait_and_NonClickbait = pd.read_csv('ro_combined_file.csv', encoding='utf-8')
print(df_Clickbait_and_NonClickbait.head())
print(df_Clickbait_and_NonClickbait['label'].value_counts())

df_Clickbait_and_NonClickbait['Headline'] = df_Clickbait_and_NonClickbait['Headline'].str.lower()
df_Clickbait_and_NonClickbait['Headline'] = df_Clickbait_and_NonClickbait['Headline'].astype(str)
df_Clickbait_and_NonClickbait['Headline'] = df_Clickbait_and_NonClickbait['Headline'].apply(lambda x: re.sub(r'[^\w\s.,!?;:ăâîșțĂÂÎȘȚ]', '', x))
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return tokens
df_Clickbait_and_NonClickbait['tokens'] = df_Clickbait_and_NonClickbait['Headline'].apply(preprocess_text)

df_Clickbait_and_NonClickbait['cleaned_Headline'] = df_Clickbait_and_NonClickbait['tokens'].apply(lambda x: ' '.join(x))

print(df_Clickbait_and_NonClickbait.head())

X_train, X_test, y_train, y_test = train_test_split(df_Clickbait_and_NonClickbait['cleaned_Headline'], df_Clickbait_and_NonClickbait['label'], test_size=0.2, random_state=42, stratify=df_Clickbait_and_NonClickbait['label'])

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_vectorized, y_train)

y_pred = svm_model.predict(X_test_vectorized)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Clickbait', 'Clickbait'], yticklabels=['Non-Clickbait', 'Clickbait'])
plt.title("Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
