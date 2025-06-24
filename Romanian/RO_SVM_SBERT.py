#!pip install sentence-transformers
#!pip install spacy
#!python -m spacy download ro_core_news_lg

import pandas as pd
import re
import spacy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
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

X_text = df_Clickbait_and_NonClickbait['cleaned_Headline']
y = df_Clickbait_and_NonClickbait['label']
X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, stratify=y, random_state=42)

#embeddings
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
X_train_embeddings = model.encode(X_train_text.tolist(), show_progress_bar=True)
X_test_embeddings = model.encode(X_test_text.tolist(), show_progress_bar=True)

#SVM
svm_model = SVC(kernel='linear', C=1.0, random_state=42, class_weight='balanced', probability=True)
svm_model.fit(X_train_embeddings, y_train)

y_pred = svm_model.predict(X_test_embeddings)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d',cmap='Blues', xticklabels=['Non-Clickbait', 'Clickbait'], yticklabels=['Non-Clickbait', 'Clickbait'])
plt.title("Confusion Matrix for SVM Model")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
