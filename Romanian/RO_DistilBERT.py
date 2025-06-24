#!pip install spacy
#!python -m spacy download ro_core_news_lg

import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from google.colab import files
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

nlp = spacy.load("ro_core_news_lg")

#uploaded = files.upload()

df_Clickbait_and_NonClickbait = pd.read_csv('ro_combined_file.csv', encoding='utf-8')
print(df_Clickbait_and_NonClickbait.head())
print(df_Clickbait_and_NonClickbait['label'].value_counts())
df_Clickbait_and_NonClickbait['Headline'] = df_Clickbait_and_NonClickbait['Headline'].fillna('')

#Data spliting
train_df, test_df = train_test_split(
    df_Clickbait_and_NonClickbait,
    test_size=0.2,
    random_state=42,
    stratify=df_Clickbait_and_NonClickbait['label']
)

#Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s.,!?;:ăâîșțĂÂÎȘȚ]', '', text)
    return text
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(tokens)

train_df['cleaned_Headline'] = train_df['Headline'].apply(clean_text).apply(preprocess_text)
test_df['cleaned_Headline'] = test_df['Headline'].apply(clean_text).apply(preprocess_text)

#Model
model_name = "racai/distilbert-base-romanian-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

max_len = 20
train_encodings = tokenizer(train_df['cleaned_Headline'].tolist(), max_length=max_len,padding='max_length', truncation=True, return_tensors='tf')
test_encodings = tokenizer(test_df['cleaned_Headline'].tolist(), max_length=max_len,padding='max_length', truncation=True, return_tensors='tf')

input_ids_train = train_encodings['input_ids'].numpy()
attention_mask_train = train_encodings['attention_mask'].numpy()
input_ids_test = test_encodings['input_ids'].numpy()
attention_mask_test = test_encodings['attention_mask'].numpy()

y_train = train_df['label'].values
y_test = test_df['label'].values

train_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids_train, 'attention_mask': attention_mask_train}, y_train)).shuffle(1000).batch(8)
test_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids_test, 'attention_mask': attention_mask_test}, y_test)).batch(8)

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

#Training
history = model.fit(train_dataset, validation_data=test_dataset, epochs=10, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)])

#Evaluation
y_pred_logits = model.predict(test_dataset).logits
y_pred = np.argmax(y_pred_logits, axis=1)

print("Train Accuracy:", model.evaluate(train_dataset, verbose=0)[1])
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Non-Clickbait', 'Clickbait']))

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')
plt.show()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Clickbait', 'Clickbait'], yticklabels=['Non-Clickbait', 'Clickbait'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.show()
