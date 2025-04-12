import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

#load the model

model = tf.keras.models.load_model('salary_pred_mode.h5')

#load the encoders and scaler from pickle
with open("label_encoder.pkl",'rb') as f:
    label_encoder = pickle.load(f)
with open("onehot.pkl",'rb') as f:
    onehot = pickle.load(f)
with open("scaler.pkl",'rb') as f:
    scaler = pickle.load(f)


st.title("Streamlit Salary Prediction using ANN")

# User input
geography = st.selectbox('Geography', onehot.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
Exited = st.selectbox('Has customer exited', [0, 1])
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [Exited]
})

geo_encoded = onehot.transform([[geography]])

geo_encoded_df = pd.DataFrame(geo_encoded.toarray(),columns=onehot.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

scaled_input = scaler.transform(input_data)

prediction = model.predict(scaled_input)

st.write(f"Predicted Salary is : {prediction[0][0]}$")
