import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the dataset
data_path = 'UCI Heart Disease Dataset.csv'
data = pd.read_csv(data_path)

# Load the trained Random Forest model
model_path = 'Random_forest_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Streamlit app
def main():
    st.title("Heart Disease Prediction App")
    
    # Sidebar options
    st.sidebar.title("Navigation")
    options = ["Explore Dataset", "Predict Heart Disease"]
    choice = st.sidebar.radio("Select an option:", options)

    if choice == "Explore Dataset":
        st.header("Dataset Overview")
        st.write(data.head())
        st.write("Shape of the dataset:", data.shape)
        st.write("Summary Statistics:")
        st.write(data.describe())

    elif choice == "Predict Heart Disease":
        st.header("Make a Prediction")
        
        # User inputs for prediction
        st.write("Provide the following inputs:")
        user_input = {}
        for col in data.columns[:-1]:  # Exclude target column
            user_input[col] = st.number_input(f"{col}:", min_value=float(data[col].min()), max_value=float(data[col].max()))
        
        # Convert inputs to a DataFrame
        input_df = pd.DataFrame([user_input])
        
        # Preprocess inputs
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_prob = model.predict_proba(input_scaled)
        
        # Display results
        st.write("Prediction (0 = No Heart Disease, 1 = Heart Disease):", int(prediction[0]))
        st.write("Prediction Probability:", prediction_prob[0])
        
if __name__ == "__main__":
    main()