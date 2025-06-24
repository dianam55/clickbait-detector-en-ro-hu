#!pip install spacy
#!python -m spacy download ro_core_news_lg
#!wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ro.300.vec.gz
#!gunzip cc.ro.300.vec.gz

import pandas as pd
import spacy
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import layers, models
from keras.optimizers import Adam
from keras.layers import Dense, LSTM, Dropout, Embedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from google.colab import files
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

nlp = spacy.load("ro_core_news_lg", disable=["parser", "ner"])

#uploaded = files.upload()
df_Clickbait_and_NonClickbait = pd.read_csv('ro_combined_file.csv', encoding='utf-8')
print(df_Clickbait_and_NonClickbait.head())
print(df_Clickbait_and_NonClickbait['label'].value_counts())

# Preprocessing
df_Clickbait_and_NonClickbait['Headline'] = df_Clickbait_and_NonClickbait['Headline'].str.lower()
df_Clickbait_and_NonClickbait['Headline'] = df_Clickbait_and_NonClickbait['Headline'].astype(str)
df_Clickbait_and_NonClickbait['Headline'] = df_Clickbait_and_NonClickbait['Headline'].apply(lambda x: re.sub(r'[^\w\s.,!?;:ăâîșțĂÂÎȘȚ]', '', x))

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop or token.text in ['!', '?', '.', ',', ';', ':']]
    return " ".join(tokens)

df_Clickbait_and_NonClickbait['cleaned_Headline'] = df_Clickbait_and_NonClickbait['Headline'].apply(preprocess_text)
print(df_Clickbait_and_NonClickbait.head())

#Train/test split
X_text = df_Clickbait_and_NonClickbait['cleaned_Headline']
y = df_Clickbait_and_NonClickbait['label'].values
X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42, stratify=y)

max_words = 5000
max_len = 30
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train_text)

X_train_sequences = tokenizer.texts_to_sequences(X_train_text)
X_test_sequences = tokenizer.texts_to_sequences(X_test_text)

X_train = pad_sequences(X_train_sequences, maxlen=max_len, padding='post', truncating='post')
X_test = pad_sequences(X_test_sequences, maxlen=max_len, padding='post', truncating='post')

#FastText embeddings
embedding_dim = 300
fasttext_path = '/content/cc.ro.300.vec'

embeddings_index = {}
with open(fasttext_path, encoding='utf-8', newline='\n', errors='ignore') as f:
    next(f)
    for line in f:
        values = line.rstrip().split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

#Embedding matrix
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

#LSTM
model = models.Sequential([
    Embedding(max_words, embedding_dim, input_length=max_len, weights=[embedding_matrix], trainable=True),
    LSTM(64, return_sequences=True),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Training
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
batch_size = 16
epochs = 30
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), callbacks=[early_stop])

# Evaluation
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
print("Train Accuracy:", model.evaluate(X_train, y_train, verbose=0)[1])
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
