import pandas as pd
import spacy
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import itertools

nlp = spacy.load("hu_core_news_lg")

df_Clickbait_and_NonClickbait = pd.read_csv('c:\\Users\\Lenovo\\Desktop\\Disertatie\\Datasets\\HU\\combined_hu_dataset.csv', encoding='utf-8')
print(df_Clickbait_and_NonClickbait.head())
print(df_Clickbait_and_NonClickbait['label'].value_counts())

#Train/test split
X = df_Clickbait_and_NonClickbait['Headline']
y = df_Clickbait_and_NonClickbait['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_df = pd.DataFrame({'Headline': X_train, 'label': y_train}).reset_index(drop=True)
test_df = pd.DataFrame({'Headline': X_test, 'label': y_test}).reset_index(drop=True)

#Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ\s!?;:.]', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(tokens)

train_df['cleaned_Headline'] = train_df['Headline'].apply(preprocess_text)
test_df['cleaned_Headline'] = test_df['Headline'].apply(preprocess_text)

#Semantic features
train_df['headline_length'] = train_df['Headline'].apply(len)
train_df['word_count'] = train_df['Headline'].apply(lambda x: len(x.split()))
train_df['exclamation_count'] = train_df['Headline'].str.count('!')
train_df['has_hyperbole'] = train_df['Headline'].str.contains('megdöbbentő|hihetetlen|csodálatos', case=False).astype(int)

test_df['headline_length'] = test_df['Headline'].apply(len)
test_df['word_count'] = test_df['Headline'].apply(lambda x: len(x.split()))
test_df['exclamation_count'] = test_df['Headline'].str.count('!')
test_df['has_hyperbole'] = test_df['Headline'].str.contains('megdöbbentő|hihetetlen|csodálatos', case=False).astype(int)

#TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
tfidf_train = tfidf_vectorizer.fit_transform(train_df['cleaned_Headline'])
tfidf_test = tfidf_vectorizer.transform(test_df['cleaned_Headline'])

scaler = StandardScaler()
semantic_features_train = train_df[['headline_length', 'word_count', 'exclamation_count', 'has_hyperbole']].values
semantic_features_test = test_df[['headline_length', 'word_count', 'exclamation_count', 'has_hyperbole']].values
semantic_features_train_scaled = scaler.fit_transform(semantic_features_train)
semantic_features_test_scaled = scaler.transform(semantic_features_test)

X_train = np.hstack([tfidf_train.toarray(), semantic_features_train_scaled])
X_test = np.hstack([tfidf_test.toarray(), semantic_features_test_scaled])

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 30],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}

keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
print(f"Total combinations: {len(combinations)}")

results = []

for params in tqdm(combinations, desc="Parameter Combinations", unit="combo"):
    rf_model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        class_weight=params['class_weight'],
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # Cross validation
    print(f"\nEvaluating parameters: {params}")
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()

    results.append({
        'params': params,
        'mean_cv_score': mean_score,
        'std_cv_score': std_score
    })

    print(f"Mean CV Accuracy: {mean_score:.4f}, Std: {std_score:.4f}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='mean_cv_score', ascending=False)
print("\nResults for Each Parameter Combination:")
print(results_df[['params', 'mean_cv_score', 'std_cv_score']].to_string(index=False))

#Training
best_params = results_df.iloc[0]['params']
best_model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    class_weight=best_params['class_weight'],
    random_state=42,
    n_jobs=-1
)
best_model.fit(X_train, y_train)

#Evaluation
y_pred = best_model.predict(X_test)
train_acc = accuracy_score(y_train, best_model.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Non-Clickbait', 'Clickbait']))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Clickbait', 'Clickbait'], yticklabels=['Non-Clickbait', 'Clickbait'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

misclassified_idx = y_test.index[y_pred != y_test]
misclassified_headlines = df_Clickbait_and_NonClickbait.loc[misclassified_idx, 'Headline']
print("Misclassified headlines:", misclassified_headlines.tolist())
