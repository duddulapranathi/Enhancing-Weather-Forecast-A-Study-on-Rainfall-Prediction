from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
le_mandal = joblib.load('le_mandal.pkl')

class WeatherInput(BaseModel):
    Mandal: str
    Min_Temp: float
    Max_Temp: float
    Min_Humidity: float
    Max_Humidity: float
    Min_Wind_Speed: float
    Max_Wind_Speed: float
    Year: int
    Month: int
    Day: int

@app.post("/predict")
async def predict_rain(input_data: WeatherInput):
    try:
        df = pd.DataFrame([{
            "Mandal_Encoded": le_mandal.transform([input_data.Mandal])[0],
            "Min Temp (°C)": input_data.Min_Temp,
            "Max Temp (°C)": input_data.Max_Temp,
            "Min Humidity (%)": input_data.Min_Humidity,
            "Max Humidity (%)": input_data.Max_Humidity,
            "Min Wind Speed (Kmph)": input_data.Min_Wind_Speed,
            "Max Wind Speed (Kmph)": input_data.Max_Wind_Speed,
            "Year": input_data.Year,
            "Month": input_data.Month,
            "Day": input_data.Day,
            "Temp_Range": input_data.Max_Temp - input_data.Min_Temp,
            "Humidity_Range": input_data.Max_Humidity - input_data.Min_Humidity,
            "Wind_Speed_Range": input_data.Max_Wind_Speed - input_data.Min_Wind_Speed
        }])

        feature_names = ['Year', 'Month', 'Day', 'Mandal_Encoded', 'Min Temp (°C)', 'Max Temp (°C)', 
                         'Min Humidity (%)', 'Max Humidity (%)', 'Min Wind Speed (Kmph)', 
                         'Max Wind Speed (Kmph)', 'Temp_Range', 'Humidity_Range', 'Wind_Speed_Range']
        df = df[feature_names]

        X_scaled = scaler.transform(df)
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]

        return {
            "prediction": int(prediction),
            "probability": round(float(probability), 3)
        }

    except Exception as e:
        return {"error": str(e)}