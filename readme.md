🚀 Problem Statement

Diabetes is one of the fastest-growing lifestyle diseases worldwide.
Early risk detection and personalized preventive guidance can significantly reduce long-term complications.

However:

Many people ignore early risk indicators

Preventive awareness is low

Health advice is often generic and not personalized

PreventAI aims to bridge this gap using Machine Learning + AI guidance.

💡 Solution Overview

PreventAI provides:

Diabetes Risk Prediction

Uses Logistic Regression model

Predicts probability of diabetes

Categorizes risk level (Low / Moderate / High)

Personalized AI Health Assistant

Generates preventive guidance

Uses user health data + risk score

Provides supportive and non-diagnostic advice

🧠 System Architecture

User → Streamlit Frontend → ML Model → Risk Probability → AI Assistant → Personalized Guidance

(See assets/architecture.png)

🔄 Workflow

Collect user health details

Preprocess input data using scaler

Predict diabetes risk probability

Categorize risk level

Generate AI-based preventive guidance

Display results to user

(See assets/workflow.png)

📊 Dataset

PIMA Indians Diabetes Dataset

Features Used:

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

🛠 Tech Stack

Python

Streamlit

Scikit-learn

Pandas / NumPy

OpenRouter API (AI Assistant)

Logistic Regression Model

📁 Project Structure
PreventAI/
│
├── app.py
├── requirements.txt
├── model/
│   ├── diabetes_model.pkl
│   └── scaler.pkl
├── data/
│   └── diabetes.csv
├── assets/
│   ├── logo.png
│   ├── architecture.png
│   └── workflow.png
└── notebooks/
    └── model_training.ipynb
⚙ How to Run Locally

Clone the repository:

git clone <your-repo-link>

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py
🔐 Disclaimer

PreventAI provides preventive health insights only.
It does not replace professional medical diagnosis or treatment.

🌍 Future Improvements

Cloud deployment

User authentication

Report upload (PDF/CSV analysis)

Advanced ML models

Mobile optimization

Doctor integration system

👨‍💻 Team

Team Name: NextCore

Theme: HealthTech
