import streamlit as st
import numpy as np
import joblib

# Trained model loading
model = joblib.load('model.pkl')

st.title('Credit Card Fraud Detection')

st.header('Enter the transaction details:')
amount = st.number_input('Amount', min_value=0.0, value=0.0)


# Creating inputs for the PCA-transformed features V1 to V28
features = []
for i in range(1, 29):
    features.append(st.number_input(f'V{i}', value=0.0))

if st.button('Predict'):
    input_features = features + [amount]  # Exclude 'Time' if it was not used in training
    
    # Reshape the input data for the model
    input_data = np.array(input_features).reshape(1, -1)
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error('This transaction is fraudulent.')
    else:
        st.success('This transaction is legitimate.')
