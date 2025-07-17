# clickbait-detector

Project Overview

This repository contains code and datasets for detecting clickbait headlines in three languages: English, Romanian, and Hungarian. Multiple machine learning and deep learning models are implemented and evaluated, including:
- Traditional models: SVM (with two variations: SBERT embeddings and TF-IDF), Random Forest
- Deep learning models: LSTM, Transformer-based models: BERT / DistilBERT

Exploratory data analysis (EDA) is also provided for each dataset, including common bigrams, punctuation patterns, and sentiment analysis.

Datasets:

The datasets used in this project contain headlines labeled as clickbait(1) or non-clickbait(0). Each dataset contains 32.000 rows of data.
- English dataset: https://www.kaggle.com/datasets/amananandrai/clickbait-dataset
- Romanian dataset:
  
  Combined from two sources:
  
  - https://www.kaggle.com/datasets/andreeaginga/clickbait?resource=download
    
  - https://github.com/dariabroscoteanu/RoCliCo
  
  The Romanian dataset includes augmented data created by me and new headlines collected and cleaned from Reddit.
  
  The scripts for data collection are included in the repository.
  
- Hungarian dataset: https://github.com/gencsimihaly/hungarian-fakenews-dataset/blob/main/hungarian-fakenews-dataset.zip

