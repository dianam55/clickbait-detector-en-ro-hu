import pandas as pd
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

df_Clickbait_and_NonClickbait = pd.read_csv('c:\\Users\\Lenovo\\Desktop\\Disertatie\\Datasets\\HU\\combined_hu_dataset.csv', encoding='utf-8')
print(df_Clickbait_and_NonClickbait.head())
print(df_Clickbait_and_NonClickbait['label'].value_counts())

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ\s!?;:.]', '', text)
    return text.strip()

df_Clickbait_and_NonClickbait['cleaned_Headline'] = df_Clickbait_and_NonClickbait['Headline'].astype(str).apply(clean_text)

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

#Evaluation
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d',cmap='Blues', xticklabels=['Non-Clickbait', 'Clickbait'], yticklabels=['Non-Clickbait', 'Clickbait'])
plt.title("Confusion Matrix for SVM Model")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
