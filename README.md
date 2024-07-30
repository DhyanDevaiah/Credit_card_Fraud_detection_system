# Credit_card_Fraud_detection_system

This project is a credit card fraud detection system that leverages machine learning and speech recognition to identify fraudulent transactions. The system uses a Support Vector Machine (SVM) model to classify transaction types based on clues provided either through text input or speech recognition.

Usage:
1) Ensure you have the dataset file fraud_datasets.file in the project directory.
2) To the main script where it only predicts fraud statements, run the below command
     python fraud.py
   Else to run the code where we can make changes to the datasets, run the below command
     python fraud_dev.py
3) The system will start listening for speech input. Speak clearly into your microphone. The system will transcribe the speech, classify the transaction type, and provide feedback on the accuracy of the classification.

Features:
1)  Machine Learning Model: Uses SVM to classify transaction types.
2)  Text Input: Accepts text clues for prediction.
3)  Speech Recognition: Transcribes speech input for prediction.
4)  Model Retraining: Allows retraining of the model with new data.

Contributing:
 Contributions are welcome! Please follow these steps to contribute:
 1) Fork the repository.
 2) Create a new branch (git checkout -b feature-branch).
 3) Make your changes and commit them (git commit -m 'Add new feature').
 4) Push to the branch (git push origin feature-branch).
 5) Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.



Feel free to reach out if you have any questions or need further assistance!

Happy coding!
