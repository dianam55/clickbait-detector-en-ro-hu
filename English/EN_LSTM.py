import pandas as pd
from google.colab import files
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import layers, models, optimizers, regularizers
from keras.optimizers import Adam
from keras.layers import Dense, LSTM, Dropout, Embedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

!wget --no-check-certificate https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
!unzip glove.6B.zip -d glove

uploaded = files.upload()

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

df_Clickbait_and_NonClickbait = pd.read_csv('clickbait_data.csv', encoding='utf-8')
df_Clickbait_and_NonClickbait['Headline'] = df_Clickbait_and_NonClickbait['Headline'].str.lower()
df_Clickbait_and_NonClickbait['Headline'] = df_Clickbait_and_NonClickbait['Headline'].apply(lambda x: re.sub(r"[^a-zA-Z\s.,!?;:]", '', x))
df_Clickbait_and_NonClickbait['tokens'] = df_Clickbait_and_NonClickbait['Headline'].apply(word_tokenize)
stop_words = set(stopwords.words('english'))
negations = {'no', 'not', 'none', 'never', "n't"}
stop_words = stop_words.difference(negations)
df_Clickbait_and_NonClickbait['tokens'] = df_Clickbait_and_NonClickbait['tokens'].apply(lambda x: [word for word in x if word not in stop_words])
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

df_Clickbait_and_NonClickbait['tokens'] = df_Clickbait_and_NonClickbait['Headline'].apply(preprocess_text)
df_Clickbait_and_NonClickbait['cleaned_Headline'] = df_Clickbait_and_NonClickbait['tokens'].apply(lambda x: ' '.join(x))
print(df_Clickbait_and_NonClickbait.head())

X_text = df_Clickbait_and_NonClickbait['cleaned_Headline']
y = df_Clickbait_and_NonClickbait['label'].values
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

#Tokenization
max_words = 5000
max_len = 20
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train_text)

X_train_sequences = tokenizer.texts_to_sequences(X_train_text) #converts each headline into a list of word indexes
X_test_sequences = tokenizer.texts_to_sequences(X_test_text)
X_train = pad_sequences(X_train_sequences, maxlen=max_len, padding='post', truncating='post') #adds zeros after the sequence if shorter than max_len
X_test = pad_sequences(X_test_sequences, maxlen=max_len, padding='post', truncating='post') #cut off words after max_len if sequence is longer

#GloVe embeddings
embedding_dim = 300
glove_dir = '/content/glove/glove.6B.300d.txt'
embeddings_index = {}
print("Loading GloVe embeddings manually...")
with open(glove_dir, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except ValueError:
            print(f"Skipping malformed line: {line.strip()}")
print(f"Loaded {len(embeddings_index)} word vectors.")

#embedding matrix
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
print(f"Embedding matrix shape: {embedding_matrix.shape}")

#LSTM
model = models.Sequential([
    layers.Embedding(max_words, embedding_dim, weights=[embedding_matrix], trainable=False),
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 32
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
