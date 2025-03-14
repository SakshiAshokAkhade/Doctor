import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset function
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("dummy_npi_data.xlsx", engine='openpyxl')
        df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

df = load_data()

# Stop execution if the dataset isn't loaded
if df is None:
    st.stop()

# Sidebar
st.sidebar.title("🔍 Doctor Survey Prediction")
st.sidebar.success("Use this tool to predict doctor availability for surveys.")

# Main title
st.markdown("# 🏥 Doctor Survey Availability Predictor")

# Preprocessing function
def preprocess_data(df):
    df = df.dropna()
    df['Login Time'] = pd.to_datetime(df['Login Time'], errors='coerce').dt.hour
    df['Logout Time'] = pd.to_datetime(df['Logout Time'], errors='coerce').dt.hour
    df.dropna(subset=['Login Time', 'Logout Time'], inplace=True)
    df['Login Time'] = df['Login Time'].astype(int)
    df['Logout Time'] = df['Logout Time'].astype(int)
    
    le_speciality = LabelEncoder()
    le_state = LabelEncoder()
    df['Speciality'] = le_speciality.fit_transform(df['Speciality'])
    df['State'] = le_state.fit_transform(df['State'])

    X = df[['Speciality', 'State', 'Login Time', 'Logout Time', 'Usage Time (mins)', 'Count of Survey Attempts']]
    y = (df['Count of Survey Attempts'] > 1).astype(int)
    
    return df, X, y, le_speciality, le_state

df, X, y, le_speciality, le_state = preprocess_data(df)

# Train Model
@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

st.markdown("## ⏳ Select Time to Predict Doctor Availability")
time_input = st.slider("Select Time (Hour)", 0, 23, 12)

# Prediction function
def predict_doctors(time_input):
    df['Login Time'] = df['Login Time'].astype(int)
    df_filtered = df[df['Login Time'] == time_input]
    
    if df_filtered.empty:
        closest_time_idx = (df['Login Time'] - time_input).abs().idxmin()
        closest_time = df.loc[closest_time_idx, 'Login Time']
        df_filtered = df[df['Login Time'] == closest_time]
        st.warning(f"⚠️ No doctors available at {time_input}:00. Trying closest available time: {closest_time}:00")
    
    if df_filtered.empty:
        st.error("❌ No doctors found for this time.")
        return pd.DataFrame(columns=['NPI'])
    
    st.markdown("### 🔍 Available Doctors")
    count = df_filtered.shape[0]
    st.metric("✅ Available Doctors", count)
    
    # Convert encoded values back to names
    df_filtered['Speciality'] = le_speciality.inverse_transform(df_filtered['Speciality'])
    df_filtered['State'] = le_state.inverse_transform(df_filtered['State'])
    
    st.dataframe(df_filtered[['NPI', 'Speciality', 'State', 'Login Time']])
    
    required_features = ['Speciality', 'State', 'Login Time', 'Logout Time', 'Usage Time (mins)', 'Count of Survey Attempts']
    X_filtered = df_filtered[required_features]
    
    if X_filtered.empty:
        st.warning("⚠️ No matching data found for the selected time.")
        return pd.DataFrame(columns=['NPI'])
    
    predictions = model.predict(X_filtered)
    df_filtered['Prediction'] = predictions
    result = df_filtered[df_filtered['Prediction'] == 1][['NPI', 'State', 'Speciality', 'Login Time']]
    
    if result.empty:
        st.error("❌ No doctors predicted to attend at this time.")
    
    return result

if st.button("🔍 Get Doctor List"):
    result = predict_doctors(time_input)
    if not result.empty:
        st.markdown("### ✅ Predicted Doctors")
        st.dataframe(result)
        
        csv = result.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download CSV", data=csv, file_name="doctor_list.csv", mime="text/csv")   
