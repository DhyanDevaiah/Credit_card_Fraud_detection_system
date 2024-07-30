from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# Load the data
try:
    df = pd.read_csv("fraud_datasets.file", sep='\t', names=['type', 'clue'], on_bad_lines='skip')
except FileNotFoundError:
    print("Error: File not found. Please make sure 'fraud_datasets.file' exists in the specified location.")
    exit()

# Preprocess the data
df.fillna("", inplace=True)  # Replace missing values with empty strings

# Convert 'type' to numerical labels (if it's categorical)
if df['type'].dtype == 'object':
    label_encoder = LabelEncoder()
    df['type_encoded'] = label_encoder.fit_transform(df['type'])
else:
    df['type_encoded'] = df['type']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['clue'], df['type_encoded'], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Support Vector Machine (SVM) model
model = SVC()
model.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)

# Function to predict type
def predict_type(new_clue):
    try:
        new_clue_vec = vectorizer.transform([new_clue])
        prediction = model.predict(new_clue_vec)[0]
        
        if df['type'].dtype == 'object':
            prediction = label_encoder.inverse_transform([prediction])[0]
        
        return prediction
    except Exception as e:
        print("Error during prediction:", e)
        return None

# Function to retrain the model with new data
def retrain_model(new_clue, correct_type):
    global df, vectorizer, model, label_encoder  # Declare to use global variables

    # Append new data to the dataframe
    new_data = pd.DataFrame({'clue': [new_clue], 'type': [correct_type]})
    df = pd.concat([df, new_data], ignore_index=True)

    # Re-encode the labels
    if df['type'].dtype == 'object':
        df['type_encoded'] = label_encoder.fit_transform(df['type'])
    else:
        df['type_encoded'] = df['type']  # Assuming 'type' is already numerical

    # Split the dataset again
    X_train, X_test, y_train, y_test = train_test_split(df['clue'], df['type_encoded'], test_size=0.2, random_state=42)

    # Vectorize the text data
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train the model
    model = SVC()
    model.fit(X_train_vec, y_train)

    print("Model retrained with new data.")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    clue = data.get('clue', '')
    predicted_type = predict_type(clue)
    return jsonify({'predicted_type': predicted_type})

@app.route('/retrain', methods=['POST'])
def retrain():
    data = request.get_json()
    clue = data.get('clue', '')
    correct_type = data.get('type', '')
    retrain_model(clue, correct_type)
    return jsonify({'status': 'Model retrained'})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
