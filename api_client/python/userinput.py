from flask import Flask, request, jsonify
from flask_cors import CORS
import xml.etree.ElementTree as ET
from cdg_client import CDGClient
from configparser import ConfigParser
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# Configuration Constants
BILL_HR = "hr"
CONGRESS = 117
API_KEY_PATH = "../secrets.ini"
BILL_NUMS = [1, 3, 4, 5, 6, 7, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 97, 105, 106, 120, 121, 122, 124, 125, 126, 127, 129, 130, 137, 138, 139, 140, 141, 143, 144]

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
    bill_info = {
        'title': bill_xml.findtext(".//title").strip(),
        'text': bill_xml.findtext(".//text"),
        'sponsor': bill_xml.findtext(".//sponsor"),
        'cosponsors': [cosponsor.text for cosponsor in bill_xml.findall(".//cosponsors/item")]
    }
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

def summarize_bill_with_gemini(bill_title):
    """Summarize the bill using the Gemini API."""
    api_key = os.getenv("GOOGLE_API_KEY")
    # Ensure the title is not empty
    if not bill_title:
        raise ValueError("Bill title is empty. Cannot summarize an empty title.")
    # Set the API key
    genai.configure(api_key=api_key)
    # Create the model
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    # Start a chat session
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    f"""Please provide an informational and educational summary of the
                    following bill titled '{bill_title}':\n\n The summary should be factual and
                    provide key details about the bill's purpose, main provisions, and any
                    significant impacts. Please do not generate any information that doesn't directly have
                    to do with the bill (such as informing me that I did not provide enough information).
                    Please search the web for the information if you can. If you are unable to, please create a
                    summary from the information you know. Please keep your response educational and precise, and style
                    it for better user readability. If there are multiple bills with the same number, list all of
                    them."""
                ],
            }
        ]
    )
    # Send a valid message
    response = chat_session.send_message("Please provide a summary.")
    return response.text

@app.route('/fetch-bill', methods=['POST'])
def fetch_bill():
    data = request.json
    congress = data.get('congress')
    bill_num = data.get('bill_num')

    if not congress or not bill_num:
        return jsonify({'error': 'Congress number and Bill number are required'}), 400

    # Validate congress and bill number
    if not congress.isdigit() or not bill_num.isdigit():
        return jsonify({'error': 'Invalid Congress number or Bill number format'}), 400

    try:
        # Initialize API client
        config = ConfigParser()
        config.read(API_KEY_PATH)
        api_key = config.get("cdg_api", "api_auth_key")
        client = CDGClient(api_key, response_format="xml")
        # Set the Google API key environment variable
        google_api_key = config.get("google_api", "api_key")
        os.environ["GOOGLE_API_KEY"] = google_api_key
        # Load the trained model
        model = joblib.load('best_model.pkl')
        # Fetch bill data
        bill_data = fetch_bill_data(client, congress, BILL_HR, [bill_num])
        if not bill_data:
            return jsonify({'error': 'No data found for the given Congress number and Bill number'}), 404
        # Use the first element of the bill_data list
        bill_info = bill_data[0]
        bill_title = bill_info['title']
        if not bill_title:
            return jsonify({'error': 'The fetched bill has no title. Cannot proceed with summarization.'}), 400
        summary = summarize_bill_with_gemini(bill_title)
        # Preprocess the text
        bill_text = preprocess_text(bill_info['text'])
        # Predict and display the result
        prediction = model.predict([bill_text])[0]
        probabilities = model.predict_proba([bill_text])[0]
        # Create a dictionary of probabilities for each class
        class_probabilities = {label: prob for label, prob in zip(model.classes_, probabilities)}
        # Return the response
        return jsonify({
            'title': bill_title,
            'summary': summary,
            'classification': class_probabilities,
            'stance': {
                'democrat': class_probabilities.get('Democrat', 0),
                'republican': class_probabilities.get('Republican', 0),
                'independent': class_probabilities.get('Independent', 0)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
    # Check if the length of the party list matches the number of rows in the DataFrame
    if len(df) != len(BILL_NUMS):
        print(f"Length of party list ({len(BILL_NUMS)}) does not match the number of rows in the DataFrame ({len(df)}). Adjusting the party list...")
        party_list = ['Democratic', 'Democratic', 'Democratic', 'Democratic', 'Democratic', 'Democratic',
                      'Republican', 'Republican', 'Middle', 'Middle', 'Middle', 'Republican',
                      'Republican', 'Middle', 'Middle', 'Republican', 'Republican', 'Democratic', 'Republican',
                      'Republican', 'Republican', 'Republican', 'Republican', 'Republican', 'Republican',
                      'Middle', 'Democratic', 'Republican', 'Middle', 'Republican', 'Republican', 'Middle',
                      'Middle', 'Middle', 'Republican', 'Middle', 'Republican', 'Democratic', 'Middle',
                      'Democratic', 'Republican', 'Middle', 'Middle', 'Republican', 'Republican', 'Middle',
                      'Middle', 'Democratic', 'Republican', 'Republican', 'Republican', 'Republican',
                      'Democratic', 'Middle', 'Middle', 'Democratic', 'Middle', 'Republican', 'Republican',
                      'Middle', 'Middle', 'Middle', 'Middle', 'Democratic', 'Middle', 'Republican', 'Middle',
                      'Democratic', 'Democratic', 'Democratic', 'Democratic', 'Democratic', 'Democratic',
                      'Democratic', 'Democratic', 'Democratic', 'Democratic', 'Democratic', 'Democratic',
                      'Democratic', 'Democratic', 'Democratic', 'Republican', 'Middle', 'Republican',
                      'Republican', 'Middle']
    else:
        party_list = ['Democratic', 'Democratic', 'Democratic', 'Democratic', 'Democratic', 'Democratic',
                      'Republican', 'Republican', 'Middle', 'Middle', 'Middle', 'Republican',
                      'Republican', 'Middle', 'Middle', 'Republican', 'Republican', 'Democratic', 'Republican',
                      'Republican', 'Republican', 'Republican', 'Republican', 'Republican', 'Republican',
                      'Middle', 'Democratic', 'Republican', 'Middle', 'Republican', 'Republican', 'Middle',
                      'Middle', 'Middle', 'Republican', 'Middle', 'Republican', 'Democratic', 'Middle',
                      'Democratic', 'Republican', 'Middle', 'Middle', 'Republican', 'Republican', 'Middle',
                      'Middle', 'Democratic', 'Republican', 'Republican', 'Republican', 'Republican',
                      'Democratic', 'Middle', 'Middle', 'Democratic', 'Middle', 'Republican', 'Republican',
                      'Middle', 'Middle', 'Middle', 'Middle', 'Democratic', 'Middle', 'Republican', 'Middle',
                      'Democratic', 'Democratic', 'Democratic', 'Democratic', 'Democratic', 'Democratic',
                      'Democratic', 'Democratic', 'Democratic', 'Democratic', 'Democratic', 'Democratic',
                      'Democratic', 'Democratic', 'Democratic', 'Republican', 'Middle', 'Republican',
                      'Republican', 'Middle']
    df['party'] = party_list
    # Preprocess data
    df['text'] = df['text'].fillna('').apply(preprocess_text)  # Preprocess the text data
    labels = df['party']
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.80, random_state=32)
    # Create and train the model with GridSearchCV for hyperparameter tuning
    pipeline_lr = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=10000))
    pipeline_rf = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100))
    # Voting Classifier
    voting_clf = VotingClassifier(estimators=[
        ('lr', pipeline_lr),
        ('rf', pipeline_rf)
    ], voting='soft')
    # GridSearchCV for Voting Classifier
    param_grid = {
        'lr__tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
        'lr__logisticregression__C': [0.1, 1, 10],
        'rf__tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
        'rf__randomforestclassifier__n_estimators': [100, 200]
    }
    grid_search = GridSearchCV(voting_clf, param_grid, cv=5)
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
    joblib.dump(best_model, 'best_model.pkl')
    app.run(debug=True)