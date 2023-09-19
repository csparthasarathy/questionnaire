import streamlit as st
import joblib
import numpy as np
from pymongo import MongoClient
from pymongo import DESCENDING

modelp = joblib.load("qmodel.pkl")

client = MongoClient("mongodb+srv://moodtrackr:moodtrackr1@moodtrackr.hwwdhdl.mongodb.net/?retryWrites=true&w=majority&appName=AtlasApp")
db = client['test']
collection = db["modelpredictions"]

def predict_recent_data():
    recent_data = collection.find().sort([('_id', DESCENDING)]).limit(1)
    
    for item in recent_data:
        del item['_id']
        del item['__v']
        l1 = item.values()
        a = list(l1)
        
        a = np.array(a).reshape(1, -1)
        pred = modelp.predict(a)
        return pred[0, 0], pred[0, 1]

# Streamlit app
st.title("Mood Prediction App")

# Button to trigger prediction
if st.button("Predict My Mood"):
    anxiety_level, stress_level = predict_recent_data()
    st.write("Your Anxiety Level:", anxiety_level)
    st.write("Your Stress Level:", stress_level)
