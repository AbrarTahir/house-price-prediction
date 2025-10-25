import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# --- Load the trained model safely ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "house_price_model.pkl")
model = joblib.load(MODEL_PATH)

# --- Streamlit App Configuration ---
st.set_page_config(page_title="🏡 House Price Predictor", layout="centered")
st.title("🏡 California House Price Prediction App")
st.write("Enter the housing details below to estimate the median house price:")

# --- Sidebar for User Inputs ---
st.sidebar.header("Input Features")

longitude = st.sidebar.number_input("Longitude", value=-122.0)
latitude = st.sidebar.number_input("Latitude", value=37.0)
housing_median_age = st.sidebar.number_input("Median Age", value=20)
total_rooms = st.sidebar.number_input("Total Rooms", value=2000)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", value=400)
population = st.sidebar.number_input("Population", value=800)
households = st.sidebar.number_input("Households", value=300)
median_income = st.sidebar.number_input("Median Income", value=3.5)

# --- Ocean proximity categorical input ---
ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

# --- One-hot encode ocean proximity ---
ocean_dummies = pd.get_dummies(pd.Series([ocean_proximity]))
ocean_dummies = ocean_dummies.reindex(
    columns=['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
    fill_value=0
)

# --- Combine numeric and categorical inputs ---
input_data = pd.DataFrame([[
    longitude, latitude, housing_median_age,
    total_rooms, total_bedrooms, population,
    households, median_income
]], columns=[
    'longitude', 'latitude', 'housing_median_age',
    'total_rooms', 'total_bedrooms', 'population',
    'households', 'median_income'
])

final_input = pd.concat([input_data, ocean_dummies], axis=1)

# --- Add engineered features (same as during training) ---
final_input['bedroom_ratio'] = final_input['total_bedrooms'] / final_input['total_rooms']
final_input['household_rooms'] = final_input['total_rooms'] / final_input['households']

# --- Ensure column order matches model training ---
feature_order = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN',
    'bedroom_ratio', 'household_rooms'
]

# Add any missing columns (failsafe)
for col in feature_order:
    if col not in final_input.columns:
        final_input[col] = 0

final_input = final_input[feature_order]

# --- Predict Button ---
if st.button("🔍 Predict House Price"):
    prediction = model.predict(final_input)[0]
    st.success(f"🏠 Estimated House Price: **${prediction:,.2f}**")

