## AI-Based Diabetes Risk Prediction & Prevention System
PreventAI is an AI-powered healthcare prototype that predicts diabetes risk using Machine Learning and provides personalized preventive guidance through an intelligent AI assistant.

Built for proactive healthcare, PreventAI combines predictive analytics with conversational AI to help users better understand and manage their health risks.

I’m genuinely proud to work on a project like PreventAI 🚀

What started as just an idea turned into a fully deployed AI-powered healthcare system focused on early diabetes risk prediction and preventive guidance.

Building something that can create real-world impact while learning Machine Learning, AI integration, deployment, and product design has been an amazing experience.

✨ “The best projects aren’t just built to impress people — they’re built to solve real problems.”

This is only Phase 1. More improvements coming soon 💡🔥

## 🚀 Features
- 🔍 Diabetes Risk Prediction
- 📊 Risk Probability Analysis
- 🤖 AI Health Assistant
- 📂 Bulk CSV Prediction Support
- 🌙 Modern Dark-Themed UI
- ☁ Cloud Deployment using Streamlit
- 📈 Personalized Preventive Guidance

## 🚀 Problem Statement

Diabetes is one of the fastest-growing lifestyle diseases worldwide.
Early risk detection and personalized preventive guidance can significantly reduce long-term complications.

However:
- Many people ignore early risk indicators
- Preventive awareness is low
- Health advice is often generic and not personalized
- PreventAI aims to bridge this gap using Machine Learning + AI guidance.

## 💡 Solution Overview

PreventAI provides:
- Diabetes Risk Prediction
- Uses Logistic Regression model
- Predicts probability of diabetes
- Categorizes risk level (Low / Moderate / High)
- Personalized AI Health Assistant
- Generates preventive guidance
- Uses user health data + risk score
- Provides supportive and non-diagnostic advice

## 🧠 System Architecture

User → Streamlit Frontend → ML Model → Risk Probability → AI Assistant → Personalized Guidance

(See assets/architecture.png)

## 🔄 Workflow

- Collect user health details
- Preprocess input data using scaler
- Predict diabetes risk probability
- Categorize risk level
- Generate AI-based preventive guidance
- Display results to user
(See assets/workflow.png)

## 📊 Dataset
PIMA Indians Diabetes Dataset

## Features Used:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

## 🛠 Tech Stack
- Python
- Streamlit
- Scikit-learn
- Pandas / NumPy
- OpenRouter API (AI Assistant)
- Logistic Regression Model

## 📁 Project Structure
```plaintext
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
```

## 🔐 Disclaimer

PreventAI provides preventive health insights only.
It does not replace professional medical diagnosis or treatment.

## 🌍 Future Improvements
- Cloud deployment
- User authentication
- Report upload (PDF/CSV analysis)
- Advanced ML models
- Mobile optimization
- Doctor integration system
