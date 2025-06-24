#!python -m spacy download ro_core_news_lg
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AdamWeightDecay
import tensorflow as tf
from google.colab import files
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import requests

nlp = spacy.load("ro_core_news_lg")

#uploaded = files.upload()
df_Clickbait_and_NonClickbait = pd.read_csv('ro_combined_file.csv', encoding='utf-8')
print(df_Clickbait_and_NonClickbait.head())
print(df_Clickbait_and_NonClickbait['label'].value_counts())

# Data splitting
train_df, test_df = train_test_split(
    df_Clickbait_and_NonClickbait,
    test_size=0.2,
    random_state=42,
    stratify=df_Clickbait_and_NonClickbait['label']
)

train_df['Headline'] = train_df['Headline'].astype(str).fillna('')
test_df['Headline'] = test_df['Headline'].astype(str).fillna('')

# Preprocessing
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

train_df['has_number'] = train_df['Headline'].apply(lambda x: 1 if re.search(r'\d+', x) else 0)
test_df['has_number'] = test_df['Headline'].apply(lambda x: 1 if re.search(r'\d+', x) else 0)
train_df['has_exclamation'] = train_df['Headline'].apply(lambda x: 1 if '!' in x else 0)
test_df['has_exclamation'] = test_df['Headline'].apply(lambda x: 1 if '!' in x else 0)

# Model
model_name = "dumitrescustefan/bert-base-romanian-cased-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

max_len = 20
train_encodings = tokenizer(train_df['cleaned_Headline'].tolist(), max_length=max_len, padding='max_length', truncation=True, return_tensors='tf')
test_encodings = tokenizer(test_df['cleaned_Headline'].tolist(), max_length=max_len, padding='max_length', truncation=True, return_tensors='tf')

input_ids_train = train_encodings['input_ids'].numpy()
attention_mask_train = train_encodings['attention_mask'].numpy()
input_ids_test = test_encodings['input_ids'].numpy()
attention_mask_test = test_encodings['attention_mask'].numpy()

y_train = train_df['label'].values
y_test = test_df['label'].values


train_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids_train, 'attention_mask': attention_mask_train}, y_train)).shuffle(1000).batch(8)
test_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids_test, 'attention_mask': attention_mask_test}, y_test)).batch(8)

optimizer = AdamWeightDecay(learning_rate=3e-5, weight_decay_rate=0.01)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=7,
    #callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)]
)

def query_llama_api(headline, api_key):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/llama-3-8b-instruct",
        "messages": [
            {"role": "system", "content": "You are an assistant that classifies Romanian headlines."},
            {"role": "user", "content": f"Classify the following Romanian headline as 'Clickbait' or 'Non-Clickbait'. Provide the label and a confidence score (0-1).\n\nHeadline: {headline}\n\nOutput format: {{\"label\": \"Clickbait\" or \"Non-Clickbait\", \"confidence\": float}}"}
        ],
        "temperature": 0.2,
        "max_tokens": 100
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        content = result['choices'][0]['message']['content']
        print("LLAMA RESPONSE:", content)
      
        match = re.search(r'"label":\s*"?(Clickbait|Non-Clickbait)"?,\s*"confidence":\s*([0-9.]+)', content)
        if match:
            label = 1 if match.group(1) == "Clickbait" else 0
            confidence = float(match.group(2))
            return label, confidence
        else:
            return None, None
    except Exception as e:
        print(f"Error querying OpenRouter LLaMA API: {e}")
        return None, None

# Post-Processing with LLaMA
clickbait_keywords = ["șocant", "incredibil", "secretul", "află", "nu vei crede", "top", "trucuri", "descoperă", "ce s-a întâmplat"]

def restore_diacritics(text):
    diacritic_map = {'s': 'ș', 't': 'ț', 'a': 'ă', 'i': 'î', 'a': 'â'}
    for wrong, correct in diacritic_map.items():
        text = text.replace(wrong + ' ', correct + ' ')
    return text

def post_process_predictions(headlines, logits, threshold=0.6, api_key=None):
    probabilities = tf.nn.softmax(logits, axis=1).numpy()
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    final_predictions = []
    uncertain_cases = []

    for i, (headline, pred, conf) in enumerate(zip(headlines, predictions, confidences)):
        headline = restore_diacritics(headline)
        if conf < threshold:
            llama_label, llama_confidence = query_llama_api(headline, api_key)
            if llama_label is not None and llama_confidence > 0.7:  # Trust LLaMA if confident
                pred = llama_label
            else:
                headline_lower = headline.lower()
                has_clickbait_keywords = any(keyword in headline_lower for keyword in clickbait_keywords)
                word_count = len(headline.split())
                is_question = '?' in headline
                has_number = bool(re.search(r'\d+', headline))
                is_clickbait_structure = is_question or has_number or len(re.findall(r'\b(top|cele|moduri)\b', headline_lower)) > 0

                if has_clickbait_keywords or (word_count < 8 and is_clickbait_structure):
                    pred = 1
                else:
                    uncertain_cases.append({
                        'headline': headline,
                        'prediction': 'uncertain',
                        'distilbert_confidence': conf,
                        'llama_confidence': llama_confidence,
                        'original_pred': pred
                    })
        final_predictions.append(pred)

    if uncertain_cases:
        pd.DataFrame(uncertain_cases).to_csv('uncertain_predictions.csv', index=False)

    return np.array(final_predictions), probabilities

api_key = "placeholder" #not a real key! keys are not shared, use your own
y_pred_logits = model.predict(test_dataset).logits
y_pred, probabilities = post_process_predictions(test_df['Headline'].tolist(), y_pred_logits, api_key=api_key)

#Evaluation
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
plt.savefig('loss_plot_ro.png')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(np.max(probabilities, axis=1), bins=20, kde=True)
plt.title('Confidence Distribution of Romanian BERT Predictions')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.savefig('confidence_distribution_ro.png')

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Clickbait', 'Clickbait'], yticklabels=['Non-Clickbait', 'Clickbait'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_ro.png')
