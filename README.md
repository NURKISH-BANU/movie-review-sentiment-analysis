🎬 Movie Review Sentiment Analysis (IMDB)
📌 Project Overview

This project focuses on building an end-to-end Sentiment Analysis system that classifies movie reviews as Positive or Negative using Natural Language Processing (NLP) and Machine Learning techniques.
The final trained model is deployed as an interactive Streamlit web application for real-time predictions.

🎯 Problem Statement

With a large number of movie reviews available online, manually understanding audience sentiment is time-consuming.
This project automates sentiment classification to help identify public opinion efficiently.

📊 Dataset

IMDB Movie Reviews Dataset

50,000 reviews

Balanced dataset:

Positive reviews

Negative reviews

⚙️ Methodology

Data Preprocessing

Removed HTML tags

Converted text to lowercase

Removed special characters and stopwords

Feature Engineering

Applied TF-IDF Vectorization to convert text into numerical features

Model Building

Trained multiple ML models:

Logistic Regression

Naive Bayes

Support Vector Machine (SVM)

Random Forest

Model Evaluation

Accuracy

Precision, Recall, F1-score

Confusion Matrix

Model Optimization

Hyperparameter tuning using GridSearchCV

Deployment

Deployed the best-performing model using Streamlit

Enables real-time sentiment prediction

📈 Results

Logistic Regression achieved the best performance with ~90%+ accuracy

The deployed app provides instant sentiment predictions for user-input reviews

🛠 Tech Stack

Python

Pandas, NumPy

Scikit-learn

Natural Language Processing (NLP)

TF-IDF Vectorizer

Streamlit

Matplotlib, Seaborn

🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/movie-review-sentiment-analysis.git
cd movie-review-sentiment-analysis

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit App
streamlit run app.py

✨ Features

Clean text preprocessing pipeline

Multiple model comparison

High accuracy sentiment prediction

User-friendly Streamlit interface

Real-time predictions

📌 Future Improvements

Deep Learning models (LSTM, BERT)

Multilingual sentiment analysis

Improved UI and visual explanations

Cloud deployment

👤 Author

Nurkish Banu
Aspiring Data Scientist | Machine Learning & Data Analytics Enthusiast
