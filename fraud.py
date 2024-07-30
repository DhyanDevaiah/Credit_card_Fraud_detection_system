import speech_recognition as sr
import threading
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os

# Load the data
try:
    df = pd.read_csv("fraud_datasets.file", sep='\t', names=['type', 'clue'], on_bad_lines='skip')
except FileNotFoundError:
    print("Error: File not found. Please make sure 'fraud_call.file' exists in the specified location.")
    exit()

# Preprocess the data
df.fillna("", inplace=True)  # Replace missing values with empty strings

if df['type'].dtype == 'object':
    from sklearn.preprocessing import LabelEncoder
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

def predict_type(new_clue):
    new_clue_vec = vectorizer.transform([new_clue])
    prediction = model.predict(new_clue_vec)[0]
    
    if df['type'].dtype == 'object':
        prediction = label_encoder.inverse_transform([prediction])[0]
    
    return prediction

def callback(recognizer, audio):
    def recognize_audio():
        try:
            text = recognizer.recognize_google(audio)
            print("You said: " + text)
            
            predicted_type = predict_type(text)
            print("Predicted type:", predicted_type)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

    threading.Thread(target=recognize_audio).start()

def main():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")

    stop_listening = recognizer.listen_in_background(microphone, callback)

    try:
        while True:
            pass
    except KeyboardInterrupt:
        stop_listening(wait_for_stop=False)
        print("Stopped listening")

if __name__ == "__main__":
    main()