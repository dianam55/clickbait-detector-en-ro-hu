import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from google.colab import files
import matplotlib.pyplot as plt
import seaborn as sns

uploaded = files.upload()

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

df_Clickbait_and_NonClickbait = pd.read_csv('clickbait_data.csv', encoding='utf-8')
print(df_Clickbait_and_NonClickbait.head())
print(df_Clickbait_and_NonClickbait['label'].value_counts())

#Preprocessing
def preprocess_text(text):
    text = str(text).lower()  
    text = re.sub(r"[^a-zA-Z\s.,!?;:]", '', text)  
    tokens = word_tokenize(text) 
    return ' '.join(tokens)  

df_Clickbait_and_NonClickbait['cleaned_Headline'] = df_Clickbait_and_NonClickbait['Headline'].apply(preprocess_text)

print("Dataset Shape:", df_Clickbait_and_NonClickbait.shape)
print("Class Distribution:", df_Clickbait_and_NonClickbait['label'].value_counts())
print("Max Headline Length:", df_Clickbait_and_NonClickbait['cleaned_Headline'].str.split().str.len().max())

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
max_len = 20

#Tokenizer
def encode_texts(texts, tokenizer, max_len):  #uses the DistilBERT tokenizer to convert texts into token IDs/pad or truncate texts/returns TensorFlow tensors 
    return tokenizer(
        texts.tolist(),
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

X = encode_texts(df_Clickbait_and_NonClickbait['cleaned_Headline'], tokenizer, max_len)
y = df_Clickbait_and_NonClickbait['label'].values

input_ids = X['input_ids'].numpy()
attention_mask = X['attention_mask'].numpy()
(input_ids_train, input_ids_test, attention_mask_train, attention_mask_test, y_train, y_test) = train_test_split(input_ids, attention_mask, y, test_size=0.2, random_state=42, stratify=y)

X_train = {'input_ids': input_ids_train, 'attention_mask': attention_mask_train}
X_test = {'input_ids': input_ids_test, 'attention_mask': attention_mask_test}

train_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': X_train['input_ids'], 'attention_mask': X_train['attention_mask']},y_train)).batch(8) #creates batches of data for training and validation.
test_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': X_test['input_ids'], 'attention_mask': X_test['attention_mask']},y_test)).batch(8)

#DistilBERT model
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

#Training
history = model.fit(train_dataset, validation_data=test_dataset, epochs=5, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)])

#Evaluation
y_pred_logits = model.predict(test_dataset).logits
y_pred = np.argmax(y_pred_logits, axis=1)
print("Train Accuracy:", model.evaluate(train_dataset, verbose=0)[1])
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Non-Clickbait', 'Clickbait']))
