import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ✅ Load dataset function
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("D:\Project_Doctor\dummy_npi_data.xlsx")
        df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

df = load_data()

# ✅ Stop execution if the dataset isn't loaded
if df is None:
    st.stop()

st.write("Dataset Loaded Successfully!")

# ✅ Preprocessing function
def preprocess_data(df):
    df = df.dropna()

    # Convert login/logout time to hour format
    df['Login Time'] = pd.to_datetime(df['Login Time'], errors='coerce').dt.hour
    df['Logout Time'] = pd.to_datetime(df['Logout Time'], errors='coerce').dt.hour
    df.dropna(subset=['Login Time', 'Logout Time'], inplace=True)  # Remove any invalid times
    df['Login Time'] = df['Login Time'].astype(int)
    df['Logout Time'] = df['Logout Time'].astype(int)

    # Encode categorical columns
    le_speciality = LabelEncoder()
    le_region = LabelEncoder()
    df['Speciality'] = le_speciality.fit_transform(df['Speciality'])
    df['Region'] = le_region.fit_transform(df['Region'])

    # ✅ Define feature set and target variable
    X = df[['Speciality', 'Region', 'Login Time', 'Logout Time', 'Usage Time (mins)', 'Count of Survey Attempts']]
    y = (df['Count of Survey Attempts'] > 1).astype(int)  # Target variable

    return X, y, le_speciality, le_region

X, y, le_speciality, le_region = preprocess_data(df)

# ✅ Train Model
@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# ✅ Streamlit UI
st.title("Doctor Survey Prediction")

# ✅ Select time input
time_input = st.slider("Select Time (Hour)", 0, 23, 12)

# ✅ Prediction function
def predict_doctors(time_input):
    # Check if the selected time exists in the dataset
    if time_input not in df['Login Time'].unique():
        st.warning(f"No doctors available at {time_input}:00.")
        return pd.DataFrame(columns=['NPI'])

    # ✅ Filter data based on selected time
    df_filtered = df[df['Login Time'] == time_input]

    if df_filtered.empty:
        st.warning("No doctors found for this time.")
        return pd.DataFrame(columns=['NPI'])

    # ✅ Debugging - Check if data is filtered correctly
    st.write("Filtered Data Preview:", df_filtered.head())

    # ✅ Apply encoding (handle unseen categories gracefully)
    try:
        df_filtered['Speciality'] = df_filtered['Speciality'].apply(
            lambda x: le_speciality.transform([x])[0] if x in le_speciality.classes_ else -1
        )
        df_filtered['Region'] = df_filtered['Region'].apply(
            lambda x: le_region.transform([x])[0] if x in le_region.classes_ else -1
        )
    except ValueError:
        st.warning("Encoding mismatch. Model was trained with different categories.")
        return pd.DataFrame(columns=['NPI'])

    # ✅ Ensure the feature columns match the model's training
    required_features = ['Speciality', 'Region', 'Login Time', 'Logout Time', 'Usage Time (mins)', 'Count of Survey Attempts']
    missing_features = [col for col in required_features if col not in df_filtered.columns]

    if missing_features:
        st.error(f"Missing columns: {missing_features}")
        return pd.DataFrame(columns=['NPI'])

    X_filtered = df_filtered[required_features]

    # ✅ Check if filtered data has any samples
    if X_filtered.empty:
        st.warning("No matching data found for the selected time.")
        return pd.DataFrame(columns=['NPI'])

    try:
        # ✅ Predict and filter results
        predictions = model.predict(X_filtered)
        df_filtered['Prediction'] = predictions
        return df_filtered[df_filtered['Prediction'] == 1][['NPI']]
    except ValueError as e:
        st.error(f"Prediction error: {e}")
        return pd.DataFrame(columns=['NPI'])

# ✅ Display prediction result when button is clicked
if st.button("Get Doctor List"):
    result = predict_doctors(time_input)
    st.write(result)

    # ✅ Allow user to download results as CSV
    csv = result.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download CSV", data=csv, file_name="doctor_list.csv", mime="text/csv")
