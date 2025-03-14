import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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
st.title("🩺 Doctor Survey Prediction")
st.write("### 👋 Welcome! Select a time to predict which doctors are most likely to attend the survey.")

# User Input
selected_hour = st.slider("Select an hour", 0, 23, 12)

# Button to trigger prediction
if st.button("🔍 Show Doctor Availability"):
    # Compute available doctors at selected hour
    available_doctors = df[df['Hour'] == selected_hour]['NPI'].nunique()
    
    # Display Results
    st.success(f"✅ Total Available Doctors at {selected_hour}:00 → **{available_doctors}**")
    
    # Plot Doctor Availability by Hour
    st.write("### 📊 Doctor Availability by Hour")
    fig, ax = plt.subplots()
    df.groupby('Hour')['NPI'].nunique().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Doctors Available")
    st.pyplot(fig)
    
    # Export as CSV
    st.download_button(
        label="📥 Download CSV",
        data=df[df['Hour'] == selected_hour][['NPI']].to_csv(index=False),
        file_name="available_doctors.csv",
        mime="text/csv",
    )
