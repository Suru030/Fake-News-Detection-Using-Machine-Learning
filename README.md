# Fake News Detection Using Machine Learning

## Overview
This project uses natural language processing and machine learning to classify news articles as real or fake. The model is trained on a dataset of news headlines and body texts labeled as real or fake.

## Features
- Text preprocessing and cleaning
- TF-IDF vectorization
- Classification using Logistic Regression
- Evaluation with accuracy, precision, recall, F1-score

## Tech Stack
- Python
- scikit-learn
- pandas
- nltk

## Getting Started
1. Clone the repository
2. Install dependencies using `pip install -r requirements.txt`
3. Run `python train_model.py` to train the classifier

## Dataset
- [Fake and real news dataset from Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

## Results
Achieved ~95% accuracy using Logistic Regression on the TF-IDF features.

## Future Work
- Use deep learning models (LSTM)
- Add a web interface using Flask
