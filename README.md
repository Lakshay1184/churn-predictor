# 💳 Customer Churn Prediction App

A machine learning-powered web application built using **Streamlit** that predicts whether a customer is likely to churn based on their banking details.

---

## 🚀 Live Demo

👉 **Try the app here:**
https://churn-predictor00.streamlit.app/

---

## 📌 Project Overview

Customer churn is a major challenge in the banking sector, directly impacting revenue and customer retention strategies.

This project leverages a **deep learning model** to analyze customer attributes and predict the likelihood of churn with high accuracy.

The application provides a **user-friendly interface** where users can input customer details and receive instant predictions along with probability scores.

---

## 🧠 Features

* Real-time churn prediction
* Clean, responsive, and interactive UI (Streamlit)
* Encoded categorical inputs (Geography, Gender)
* Scaled numerical features for better model performance
* Probability-based output with clear visual feedback

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Frontend / UI:** Streamlit
* **Machine Learning:** TensorFlow / Keras
* **Data Processing:** NumPy, Pandas
* **Preprocessing:** Scikit-learn
* **Model Storage:** Pickle / JSON

---

## 📂 Project Structure

```
├── app.py                  # Main Streamlit application
├── model/
│   ├── churn_model.keras   # Trained deep learning model
│   ├── scaler.pkl         # Feature scaler
│   ├── encoder.pkl        # Label encoders
│   └── feature_config.json # Feature structure
├── data/
│   └── dataset.csv        # Dataset used for training
├── requirements.txt       # Project dependencies
├── runtime.txt            # Python version specification
└── README.md              # Project documentation
```

---

## ⚙️ How It Works

1. User inputs customer details (age, balance, geography, etc.)
2. Data is:

   * Encoded (categorical variables)
   * Scaled (numerical features)
3. Processed data is passed into the trained model
4. Model outputs:

   * Churn probability
   * Final prediction (Churn / Not Churn)

---

## 📦 Installation & Setup

Clone the repository:

```
git clone <your-repo-link>
cd churn-predictor
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the app:

```
streamlit run app.py
```

---

## 🚀 Deployment

This app is deployed using **Streamlit Cloud** and can be accessed here:

👉 https://churn-predictor00.streamlit.app/

---

## 📈 Future Improvements

* Add model explainability (SHAP / LIME)
* Improve UI with advanced visualizations
* Add user authentication
* Support batch predictions via CSV upload
