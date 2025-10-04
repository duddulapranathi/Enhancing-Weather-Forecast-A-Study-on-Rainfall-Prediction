import streamlit as st
import joblib
import pandas as pd
from datetime import datetime, timedelta

# Load model and encoders
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
le_mandal = joblib.load('le_mandal.pkl')

# Custom CSS for UI styling with background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1507525428034-b723cf961d3e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stSelectbox, .stSlider {
        font-size: 18px;
        height: 50px;
        width: 400px;
        margin-bottom: 20px;
        background-color: rgba(255, 255, 255, 0.9); /* Semi-transparent white background for readability */
        border-radius: 5px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #4a6fa5;
        color: white;
        padding: 10px 30px;
        font-size: 18px;
        border: none;
        border-radius: 5px;
        width: 400px;
    }
    .stButton>button:hover {
        background-color: #3a5f95;
    }
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent container for content */
        padding: 20px;
        border-radius: 10px;
    }
    h1 {
        color: #333333;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Hyderabad Rainfall Prediction")

# UI Inputs
mandal = st.selectbox("Mandal", le_mandal.classes_)
temp_range = st.slider("Temperature Range (°C)", 0.0, 30.0, 15.0)
humidity_range = st.slider("Humidity Range (%)", 0.0, 80.0, 40.0)
wind_speed_range = st.slider("Wind Speed Range (Kmph)", 0.0, 20.0, 10.0)

# Date selection
current_date = datetime(2025, 8, 4)  # Current date: August 4, 2025
dates = [current_date + timedelta(days=i) for i in range(7)]  # Next 7 days
date_options = [d.strftime("%d-%b-%y") for d in dates]
selected_date = st.selectbox("Date", date_options, index=0)
date_obj = pd.to_datetime(selected_date, format='%d-%b-%y')
year = date_obj.year
month = date_obj.month
day = date_obj.day

if st.button("Predict Rainfall"):
    try:
        # Set default min values and compute max values based on ranges
        min_temp = 20.0  # Default value for Min Temp (°C)
        max_temp = min_temp + temp_range
        min_humidity = 40.0  # Default value for Min Humidity (%)
        max_humidity = min_humidity + humidity_range
        min_wind_speed = 5.0  # Default value for Min Wind Speed (Kmph)
        max_wind_speed = min_wind_speed + wind_speed_range

        input_df = pd.DataFrame([{
            "Mandal_Encoded": le_mandal.transform([mandal])[0],
            "Min Temp (°C)": min_temp,
            "Max Temp (°C)": max_temp,
            "Min Humidity (%)": min_humidity,
            "Max Humidity (%)": max_humidity,
            "Min Wind Speed (Kmph)": min_wind_speed,
            "Max Wind Speed (Kmph)": max_wind_speed,
            "Year": year,
            "Month": month,
            "Day": day,
            "Temp_Range": temp_range,
            "Humidity_Range": humidity_range,
            "Wind_Speed_Range": wind_speed_range
        }])

        feature_names = ['Year', 'Month', 'Day', 'Mandal_Encoded', 'Min Temp (°C)', 'Max Temp (°C)', 
                         'Min Humidity (%)', 'Max Humidity (%)', 'Min Wind Speed (Kmph)', 
                         'Max Wind Speed (Kmph)', 'Temp_Range', 'Humidity_Range', 'Wind_Speed_Range']
        input_df = input_df[feature_names]

        X_scaled = scaler.transform(input_df)
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]

        st.success(f"Rain Prediction: {'Yes' if prediction else 'No'}")
        st.info(f"Probability of Rain: {probability:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")