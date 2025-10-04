#  Enhancing Weather Forecast: A Study on Rainfall Prediction

**_Predicting rainfall trends using Machine Learning and Deep Learning for smarter weather forecasting._**



##  Objective
The objective of this project is to develop a **machine learning and deep learning-based rainfall prediction system** for the **Hyderabad region**.  
The project aims to forecast **rain occurrence and intensity** using historical weather data to support applications in **agriculture, urban planning, and disaster management**.

Key goals include:
- Building predictive models to classify rainfall occurrence (rain/no rain).  
- Comparing the performance of ML and DL models.  
- Deploying the best-performing model using **FastAPI** and **Streamlit** for real-time predictions.



##  Workflow Overview

1. **Data Collection** – Hyderabad Weather Dataset (2023–2025) from Kaggle.  
2. **Data Preprocessing** – Cleaning, feature extraction, encoding, and scaling.  
3. **Exploratory Data Analysis (EDA)** – Understanding distributions, correlations, and rainfall patterns.  
4. **Model Development** – Applying ML and DL algorithms for rainfall prediction.  
5. **Model Evaluation** – Using metrics like Accuracy, Precision, Recall, F1-score, and ROC-AUC.  
6. **Deployment** – Implementing a FastAPI REST service and a Streamlit web dashboard for end users.

---

##  Dataset Description

- **Source:** [Hyderabad Weather Dataset (Kaggle)](https://www.kaggle.com/datasets/gouthamsunny/hyderabad-weather-dataset-2023-2025)  

Derived features include:
- `Temp_Range = MaxTemp - MinTemp`  
- `Humidity_Range = MaxHumidity - MinHumidity`  
- `Wind_Speed_Range = MaxWindSpeed - MinWindSpeed`  
- `Rain_binary` = 1 (Rain) or 0 (No Rain)



##  Data Preprocessing

- **Datetime conversion:** Converted `Date` to datetime format.  
- **Encoding:** Label encoded categorical features (e.g., `Mandal`).  
- **Feature scaling:** Standardized numeric values using `StandardScaler`.  
- **Handling class imbalance:** Used `compute_class_weight('balanced')` to give equal weight to rare rain days.  
- **Data splitting:**  
  - Train: 80%  
  - Test: 20% (Stratified sampling to maintain rain/no-rain ratio)


##  Exploratory Data Analysis (EDA)

EDA was performed using **Matplotlib**, **Seaborn**, and **Plotly** for visual insights.

### Key Analyses:
- **Distribution plots:** Rainfall and humidity distributions across mandals.  
- **Correlation matrix:** Identified strong correlation between humidity and rainfall.  
- **Seasonal trends:** Higher rainfall frequency between June and September.  
- **Visuals used:** Pie charts, histograms, scatter plots, and heatmaps for mandal-wise rainfall patterns.



## 🤖 Machine Learning Models Used

1. **Logistic Regression** – Baseline model for binary classification.  
2. **K-Nearest Neighbors (KNN)** – Non-parametric model for local pattern analysis.  
3. **Decision Tree** – Rule-based classification.  
4. **Random Forest** – Ensemble model improving stability and accuracy.  
5. **XGBoost** – Gradient boosting algorithm with best performance.



## 🧠 Deep Learning Models Used

1. **Long Short-Term Memory (LSTM)** – Captures sequential and temporal dependencies.  
2. **Recurrent Neural Network (RNN)** – Simplified time-series modeling architecture.

Both DL models were implemented using **TensorFlow/Keras** with:
- 64-unit recurrent layers  
- 32-unit dense layers (ReLU)  
- Sigmoid output layer for binary classification  
- Adam optimizer and binary cross-entropy loss



##  Model Training and Testing

- **Split:** 80% training, 20% testing  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC  
- **Cross-validation:** Ensured consistent model performance  
- **Handling Imbalance:** Applied class weights during training



##  Result Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|-----------|------------|---------|-----------|----------|
| Logistic Regression | 0.80 | 0.55 | 0.86 | 0.67 | 0.90 |
| KNN | 0.87 | 0.73 | 0.69 | 0.71 | 0.90 |
| Decision Tree | 0.87 | 0.72 | 0.71 | 0.72 | 0.81 |
| Random Forest | 0.90 | 0.85 | 0.71 | 0.77 | 0.96 |
| **XGBoost** | **0.92** | **0.85** | **0.79** | **0.82** | **0.97** |
| LSTM | 0.85 | 0.70 | 0.67 | 0.68 | 0.91 |
| RNN | 0.83 | 0.68 | 0.65 | 0.66 | 0.89 |

**Best Model:** XGBoost (Accuracy: 92%, ROC-AUC: 0.97)



##  Model Deployment

###  FastAPI Backend

- Built a **RESTful API** using FastAPI for serving real-time predictions.  
- Endpoint `/predict` accepts JSON input (temperature, humidity, wind speed, mandal, date).  
- Returns:
  {
    "prediction": "Rain Likely",
    "probability": 0.82
  }
- Serialized model using `joblib` for fast inference.  
- Hosted locally or deployable to cloud (AWS / Render / Azure).

### 🖥️ Streamlit Dashboard

An interactive user interface for end users to test rainfall predictions easily.

**Features:**
- Dropdown for `Mandal` selection.  
- Sliders for temperature, humidity, and wind speed ranges.  
- Date picker for short-term forecasting.  
- Real-time rainfall probability display.  
- Visual graphs and probability charts.

Run locally:
```
streamlit run dashboard/app.py
```

##  Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, TensorFlow, Streamlit, FastAPI, Joblib  
- **IDE:** Jupyter Notebook, VS Code  
- **Deployment Tools:** Uvicorn, Streamlit Cloud / Render  
- **Version Control:** Git & GitHub



##  Future Enhancements
- Integrate live weather data via **OpenWeather API**.  
- Extend predictions to **multi-class rainfall intensity levels**.  
- Deploy full system on **Streamlit Cloud** for public access.  
- Automate model retraining with new weather data.


