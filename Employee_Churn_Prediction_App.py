import streamlit as st
import pandas as pd
from joblib import load
import dill

# Load the pretrained model
with open('pipeline.pkl', 'rb') as file:
    model = dill.load(file)

my_feature_dict = load('my_feature_dict.pkl')

# Function to predict churn
def predict_churn(data):
    # Convert categorical columns to numeric using one-hot encoding
    data_encoded = pd.get_dummies(data, drop_first=True)
    # Predict churn
    prediction = model.predict(data_encoded)
    return prediction

st.title('Employee Churn Prediction App')
st.subheader('Based on Employee Dataset')

# Display categorical features
st.subheader('Categorical Features')
categorical_input = my_feature_dict.get('CATEGORICAL')
categorical_input_vals = {}
for i, col in enumerate(categorical_input.get('Column Name')):
    categorical_input_vals[col] = st.selectbox(col, categorical_input.get('Members')[i])

# Display numerical features
st.subheader('Numerical Features')
numerical_input = my_feature_dict.get('NUMERICAL')
numerical_input_vals = {}
for col in numerical_input.get('Column Name'):
    numerical_input_vals[col] = st.number_input(col)

# Combine numerical and categorical input dicts
input_data = pd.DataFrame.from_dict({**categorical_input_vals, **numerical_input_vals}, orient='index').T

# Churn Prediction
if st.button('Predict'):
    prediction = predict_churn(input_data)[0]
    translation_dict = {1: "Expected", 0: "Not Expected"}
    prediction_translate = translation_dict.get(prediction)
    st.write(f'The Prediction is **{prediction}**, Hence customer is **{prediction_translate}** to churn.')

st.subheader('Created by Saeeda Fatima')
