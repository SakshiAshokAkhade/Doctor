import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset with caching
@st.cache_data
def load_data():
    file_path = "dummy_npi_data.xlsx"
    df = pd.read_excel(file_path, sheet_name="Dataset")
    df['Login Time'] = pd.to_datetime(df['Login Time'])
    df['Logout Time'] = pd.to_datetime(df['Logout Time'])
    df['Hour'] = df['Login Time'].dt.hour
    
    # Encoding categorical variables
    label_enc_region = LabelEncoder()
    label_enc_speciality = LabelEncoder()
    label_enc_state = LabelEncoder()
    df['Region'] = label_enc_region.fit_transform(df['Region'])
    df['Speciality'] = label_enc_speciality.fit_transform(df['Speciality'])
    df['State'] = label_enc_state.fit_transform(df['State'])
    
    return df, label_enc_region, label_enc_speciality, label_enc_state

df, label_enc_region, label_enc_speciality, label_enc_state = load_data()

# Train and save model only once
@st.cache_resource
def train_model():
    X = df[['Hour', 'Usage Time (mins)', 'Region', 'State', 'Speciality', 'Count of Survey Attempts']]
    y = df['NPI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)  # Reduced n_estimators for speed
    model.fit(X_train, y_train)
    return model

model = train_model()

# Streamlit UI
st.title("Doctor Survey Prediction")
st.write("Enter a time to predict which doctors are most likely to attend the survey.")

# User Input
selected_hour = st.slider("Select an hour", 0, 23, 12)

# Predict function
def predict_doctors(hour):
    X_input = np.array([[hour, df['Usage Time (mins)'].mean(), 0, 0, 0, df['Count of Survey Attempts'].mean()]])
    return model.predict(X_input)

predicted_np_ids = predict_doctors(selected_hour)

# Display Results
st.write("### Predicted NPIs:")
st.write(predicted_np_ids)

# Export as CSV
output_df = pd.DataFrame({"NPI": predicted_np_ids})
st.download_button(
    label="Download CSV",
    data=output_df.to_csv(index=False),
    file_name="predicted_doctors.csv",
    mime="text/csv",
)
