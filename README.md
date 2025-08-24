Customer Churn Prediction with Neural Networks
This project uses an Artificial Neural Network (ANN) to predict customer churn — whether a bank’s customer is likely to leave (churn) or stay. The model is trained on the Customer Churn Modelling dataset and enhanced with visualizations, evaluation metrics, and prediction functions.
The main file is nuralnetwork.ipynb file you can go through other files too which i maded to learn about nurealnetworks
What This Project Does:-
This project builds and trains a Neural Network to predict customer churn — whether a bank’s customer will stay or leave.
Takes input data from the Customer Churn Modelling dataset (customer details like credit score, geography, gender, age, balance, number of products, etc.).
Cleans and preprocesses the data (encoding categorical values, scaling features).
Trains an Artificial Neural Network (ANN) using TensorFlow/Keras.
Evaluates the model with metrics such as accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.
Visualizes training performance (accuracy & loss over epochs) and churn distribution.
Saves the model so it can be reused without retraining.
Predicts churn for new customers when their details are given.

Features:-
Exploratory Data Analysis (EDA): churn distribution, feature correlations.
Neural Network Model: built with TensorFlow/Keras.
Evaluation: accuracy, confusion matrix, precision, recall, F1-score, ROC-AUC.
Training Performance: accuracy/loss curve visualization.
Model Saving & Loading: reuse the trained ANN without retraining.
Custom Prediction Function: input customer details to predict churn.

Tech Stack:-
Python
Pandas, NumPy
Scikit-learn

TensorFlow/Keras

Matplotlib, Seaborn
