#!/usr/bin/env python3
"""
   Bill Stance Classification


   This script uses the Library of Congress API to collect bill data and train a
   machine learning model to classify the bills as having a Republican, Democratic, or Middle stance.


   @copyright: 2022, Library of Congress
   @license: CC0 1.0
"""

import xml.etree.ElementTree as ET
from cdg_client import CDGClient  # Ensure this module is in your path
from configparser import ConfigParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
BILL_HR = "hr"
CONGRESS = 117
API_KEY_PATH = "../secrets.ini"
BILL_NUMS = [1, 30, 3, 4, 5, 6, 7, 31, 18, 19, 32, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
             54, 55]  # Example bill numbers to fetch

# Preprocessing Functions
def preprocess_text(text):
    """Clean and preprocess text data."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    text = text.strip()  # Remove leading and trailing whitespace
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]  # Lemmatization
    return ' '.join(words)

def get_bill_details(client, congress, bill_type, bill_num):
    """Fetch bill details from the Library of Congress API."""
    endpoint = f"bill/{congress}/{bill_type}/{bill_num}"
    data, _ = client.get(endpoint)
    return ET.fromstring(data)

def extract_bill_info(bill_xml):
    """Extract relevant information from the bill XML."""
    def get_text_or_default(element, default="0"):
        return element.text.strip() if element is not None and element.text else default

    bill_info = {
        'title': get_text_or_default(bill_xml.find(".//title")),
        'text': get_text_or_default(bill_xml.find(".//text")),
        'sponsor': get_text_or_default(bill_xml.find(".//sponsor")),
        'cosponsors': [cosponsor.text.strip() for cosponsor in bill_xml.findall(".//cosponsors/item") if cosponsor.text],
        'stance': {
            'democrat': get_text_or_default(bill_xml.find(".//stance/democrat")),
            'republican': get_text_or_default(bill_xml.find(".//stance/republican")),
            'independent': get_text_or_default(bill_xml.find(".//stance/independent"))
        }
    }
    print('Extracted bill info:', bill_info)  # Add this line to log the extracted bill info
    return bill_info

def fetch_bill_data(client, congress, bill_type, bill_nums):
    """Fetch data for multiple bills."""
    bills = []
    for bill_num in bill_nums:
        bill_xml = get_bill_details(client, congress, bill_type, bill_num)
        bill_info = extract_bill_info(bill_xml)
        bills.append(bill_info)
        print(f"Fetched Bill Title: {bill_info['title']}")  # Display the title of each bill
    return bills

if __name__ == "__main__":
    # Initialize API client
    config = ConfigParser()
    config.read(API_KEY_PATH)
    api_key = config.get("cdg_api", "api_auth_key")
    client = CDGClient(api_key, response_format="xml")

    # Fetch bill data
    print(f"Fetching bill data for Congress {CONGRESS}...")
    bill_data = fetch_bill_data(client, CONGRESS, BILL_HR, BILL_NUMS)

    # Create a DataFrame
    df = pd.DataFrame(bill_data)
    df['party'] = [
        'Democratic', 'Democratic', 'Democratic', 'Democratic', 'Democratic',
        'Democratic', 'Democratic', 'Republican', 'Republican', 'Republican',
        'Republican', 'Middle', 'Middle', 'Middle', 'Middle',
        'Republican', 'Republican', 'Middle', 'Republican', 'Republican',
        'Republican', 'Republican', 'Republican', 'Republican', 'Republican',
        'Republican', 'Middle', 'Democratic', 'Middle', 'Middle',
        'Republican', 'Middle', 'Middle', 'Middle', 'Middle',
        'Republican', 'Middle', 'Middle', 'Democratic', 'Middle',
        'Middle', 'Middle', 'Middle'
    ]  # Example labels including "Middle"

    # Preprocess data
    df['text'] = df['text'].fillna('').apply(preprocess_text)  # Preprocess the text data
    labels = df['party']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.80, random_state=32)

    # Create and train the model with GridSearchCV for hyperparameter tuning
    pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=10000))
    param_grid = {
        'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
        'logisticregression__C': [0.1, 1, 10]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Predict and evaluate
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    # Cross-Validation
    cv_scores = cross_val_score(best_model, df['text'], labels, cv=5)
    print(f'Cross-Validation Scores: {cv_scores}')
    print(f'Average Cross-Validation Score: {np.mean(cv_scores)}')

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions, labels=['Democratic', 'Republican', 'Middle'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Democratic', 'Republican', 'Middle'], yticklabels=['Democratic', 'Republican', 'Middle'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    print(f'Accuracy: {accuracy}')
    print(f'Classification Report:\n{report}')