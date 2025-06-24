import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import itertools

#uploaded = files.upload()
df_Clickbait_and_NonClickbait = pd.read_csv('clickbait_data.csv', encoding='utf-8')
print(df_Clickbait_and_NonClickbait.head())
print(df_Clickbait_and_NonClickbait['label'].value_counts())


X = df_Clickbait_and_NonClickbait['Headline']
y = df_Clickbait_and_NonClickbait['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_df = pd.DataFrame({'Headline': X_train, 'label': y_train}).reset_index(drop=True)
test_df = pd.DataFrame({'Headline': X_test, 'label': y_test}).reset_index(drop=True)

#Preprocessing
def preprocess(df_Clickbait_and_NonClickbait):
    df_Clickbait_and_NonClickbait['cleaned_Headline'] = df_Clickbait_and_NonClickbait['Headline'].str.lower().str.replace(r"[^a-zA-Z\s.,!?;:]", '', regex=True)
    #Semantic features
    df_Clickbait_and_NonClickbait['headline_length'] = df_Clickbait_and_NonClickbait['Headline'].apply(len)
    df_Clickbait_and_NonClickbait['word_count'] = df_Clickbait_and_NonClickbait['Headline'].apply(lambda x: len(x.split()))
    df_Clickbait_and_NonClickbait['exclamation_count'] = df_Clickbait_and_NonClickbait['Headline'].str.count('!')
    df_Clickbait_and_NonClickbait['has_hyperbole'] = df_Clickbait_and_NonClickbait['Headline'].str.contains('shocking|believe|amazing', case=False).astype(int)
    return df_Clickbait_and_NonClickbait

train_df = preprocess(train_df)
test_df = preprocess(test_df)

#SBERT embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
train_embeddings = model.encode(train_df['cleaned_Headline'].tolist(), show_progress_bar=True)
test_embeddings = model.encode(test_df['cleaned_Headline'].tolist(), show_progress_bar=True)

#Semantic features
train_semantic_features = train_df[['headline_length', 'word_count', 'exclamation_count', 'has_hyperbole']].values
test_semantic_features = test_df[['headline_length', 'word_count', 'exclamation_count', 'has_hyperbole']].values

scaler = StandardScaler()
train_semantic_features_scaled = scaler.fit_transform(train_semantic_features)
test_semantic_features_scaled = scaler.transform(test_semantic_features)

X_train = np.hstack([train_embeddings, train_semantic_features_scaled])
X_test = np.hstack([test_embeddings, test_semantic_features_scaled])

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

#cross-validation
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

#best model
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
