# clickbait-detector

## Project Overview

This repository contains code and datasets for detecting clickbait headlines in three languages: English, Romanian, and Hungarian. Multiple machine learning and deep learning models are implemented and evaluated, including:
- Traditional models: SVM (with two variations: SBERT embeddings and TF-IDF), Random Forest
- Deep learning models: LSTM, Transformer-based models: BERT / DistilBERT

Exploratory data analysis (EDA) is also provided for each dataset, including common bigrams, punctuation patterns, and sentiment analysis.

## Transformer Models

- English: distilbert-base-uncased
- Romanian: dumitrescustefan/bert-base-romanian-cased-v1 and racai/distilbert-base-romanian-cased
- Hungarian: SZTAKI-HLT/hubert-base-cc

## SBERT Models

- English: all-MiniLM-L6-v2
- Romanian and Hungarian: paraphrase-multilingual-MiniLM-L12-v2


## Datasets

The datasets used in this project contain headlines labeled as clickbait(1) or non-clickbait(0). The data has two variables: headline and label. Each dataset contains 32.000 rows of data.
- English dataset: https://www.kaggle.com/datasets/amananandrai/clickbait-dataset
- Romanian dataset:
  
  Combined from two sources:
  
  - https://www.kaggle.com/datasets/andreeaginga/clickbait?resource=download
    
  - https://github.com/dariabroscoteanu/RoCliCo
  
  The Romanian dataset includes augmented data (through transtalion in English and back in Romanian), new headlines collected and cleaned from Reddit and generated data using a python script.

  | Source | Num of data |
  |----------|----------|
  | Kaggle  | 10640  | 
  | RoCliCo  | 5348 | 
  | Augmented  | 7591 | 
  | Reddit  | 4000  | 
  | Generated  | 4421  | 

  
  The scripts for data collection are included in the repository.
  
- Hungarian dataset: https://github.com/gencsimihaly/hungarian-fakenews-dataset/blob/main/hungarian-fakenews-dataset.zip

