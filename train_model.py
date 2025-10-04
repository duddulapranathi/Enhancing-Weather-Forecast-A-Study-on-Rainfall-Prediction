import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import joblib

def prepare_data(data_path):
    df = pd.read_csv(data_path)
    df['Rain_binary'] = (df['Rain (mm)'] > 0).astype(int)
    df = df.drop('Rain (mm)', axis=1)
    
    le_mandal = LabelEncoder()
    df['Mandal_Encoded'] = le_mandal.fit_transform(df['Mandal'])
    
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Temp_Range'] = df['Max Temp (°C)'] - df['Min Temp (°C)']
    df['Humidity_Range'] = df['Max Humidity (%)'] - df['Min Humidity (%)']
    df['Wind_Speed_Range'] = df['Max Wind Speed (Kmph)'] - df['Min Wind Speed (Kmph)']
    df = df.drop(['Date', 'District', 'Mandal'], axis=1)
    
    feature_names = ['Year', 'Month', 'Day', 'Mandal_Encoded', 'Min Temp (°C)', 'Max Temp (°C)', 
                     'Min Humidity (%)', 'Max Humidity (%)', 'Min Wind Speed (Kmph)', 
                     'Max Wind Speed (Kmph)', 'Temp_Range', 'Humidity_Range', 'Wind_Speed_Range']
    X = df[feature_names]
    y = df['Rain_binary']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, le_mandal

def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = XGBClassifier(random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Model Precision: {precision:.2f}")
    print(f"Model Recall: {recall:.2f}")
    print(f"Model F1 Score: {f1:.2f}")
    print(f"Model AUC-ROC: {auc:.2f}")
    
    return model

if __name__ == "__main__":
    data_path = "C:\\Users\\User\\Desktop\\Hyderabad .csv"
    X_scaled, y, scaler, le_mandal = prepare_data(data_path)
    model = train_and_evaluate_model(X_scaled, y)
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le_mandal, 'le_mandal.pkl')