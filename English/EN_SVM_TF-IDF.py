#!pip install nltk

import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('stopwords')

uploaded = files.upload()
df_Clickbait_and_NonClickbait = pd.read_csv('clickbait_data.csv', encoding='utf-8')
print(df_Clickbait_and_NonClickbait.head())
print(df_Clickbait_and_NonClickbait['label'].value_counts())

#Preprocessing
df_Clickbait_and_NonClickbait['Headline'] = df_Clickbait_and_NonClickbait['Headline'].str.lower()
df_Clickbait_and_NonClickbait['Headline'] = df_Clickbait_and_NonClickbait['Headline'].apply(lambda x: re.sub(r"[^a-zA-Z\s.,!?;:]", '', x))
stop_words = set(stopwords.words('english'))
negations = {'no', 'not', 'none', 'never', "n't"}  
stop_words = stop_words.difference(negations)  #keep negation
lemmatizer = WordNetLemmatizer()
df_Clickbait_and_NonClickbait['tokens'] = df_Clickbait_and_NonClickbait['Headline'].apply(word_tokenize)
df_Clickbait_and_NonClickbait['tokens'] = df_Clickbait_and_NonClickbait['tokens'].apply(lambda x: [word for word in x if word not in stop_words])
df_Clickbait_and_NonClickbait['tokens'] = df_Clickbait_and_NonClickbait['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
df_Clickbait_and_NonClickbait['cleaned_Headline'] = df_Clickbait_and_NonClickbait['tokens'].apply(lambda x: ' '.join(x))

X_text = df_Clickbait_and_NonClickbait['cleaned_Headline']
y = df_Clickbait_and_NonClickbait['label']
X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42, stratify=y)

#TF-IDF 
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  
X_train = tfidf_vectorizer.fit_transform(X_train_text)
X_test = tfidf_vectorizer.transform(X_test_text)

param_grid = {
    'C': [0.1, 1.0, 10.0],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}

svm_model = SVC(random_state=42, class_weight='balanced')
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

#Best model
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

#Evaluation
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Clickbait', 'Clickbait'], yticklabels=['Non-Clickbait', 'Clickbait'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
