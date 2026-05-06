# ☀️ SolarAI — AI Powered Solar Power Output Prediction System

An advanced Applied Machine Learning project that predicts solar power generation using real-world weather and solar plant data.

SolarAI combines multiple machine learning models including Support Vector Regression (SVR), Random Forest Bagging, Ridge Time-Series Regression, Ensemble Learning, LDA, and K-Means Clustering to achieve highly accurate solar energy forecasting.

---

# 🚀 Features

- 🔥 Real-world solar power dataset integration
- 📈 Multi-model machine learning prediction system
- ⚡ Ensemble learning for improved accuracy
- 📊 Interactive analytics dashboard
- 🌙 Modern dark/light responsive UI
- 🧠 Feature engineering with cyclic time encoding
- 📉 Residual and regression evaluation plots
- 🌍 Real-world renewable energy forecasting use case
- 🧪 Cross-validation and outlier detection
- 🖥️ Flask-powered backend API

---

# 🧠 Machine Learning Models Used

| Model | Purpose |
|---|---|
| SVM Regression (SVR) | Captures non-linear irradiance patterns |
| Random Forest Bagging | Reduces variance and improves robustness |
| Ridge Regression | Handles time-series forecasting |
| Ensemble Learning | Combines predictions for stability |
| LDA | Dimensionality reduction |
| K-Means | Clustering similar solar conditions |

---

# 📂 Dataset

This project uses the **Kaggle India Solar Power Generation Dataset**.

Dataset includes:
- Solar power generation data
- Weather sensor data
- Irradiance
- Temperature
- Humidity
- Time-series measurements

Dataset source:
https://www.kaggle.com/datasets/anikannal/solar-power-generation-data

---

# 📊 Evaluation Metrics

The project evaluates models using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- Cross Validation
- Residual Analysis

---

# 📈 Model Performance

| Model | R² Score |
|---|---|
| SVM (SVR) | 0.9775 |
| Bagging Ensemble | 0.9860 |
| Time-Series Regression | 0.9946 |

---

# 🖼️ Evaluation Plots

The project generates:

- Actual vs Predicted Plot
- Residual Distribution Plot
- Feature Importance Visualization
- Time-Series Prediction Graphs

These plots help validate model accuracy and error behavior.

---

# ⚙️ Technologies Used

## Backend
- Python
- Flask
- scikit-learn
- NumPy
- Pandas

## Frontend
- HTML5
- CSS3
- JavaScript
- Chart.js

---

# 🏗️ Project Structure

```bash
SOLAR/
│
├── app.py
├── train.py
├── evaluate.py
├── requirements.txt
│
├── model/
│   ├── ridge_model.pkl
│   ├── scaler.pkl
│   ├── lda.pkl
│   ├── metrics.json
│   ├── residual_data.json
│   ├── cv_results.json
│   └── plots/
│       ├── actual_vs_predicted.png
│       └── residual_distribution.png
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── Plant_1_Generation_Data.csv
└── Plant_1_Weather_Sensor_Data.csv
