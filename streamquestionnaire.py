import streamlit as st
import joblib
import numpy as np
from pymongo import MongoClient
from pymongo import DESCENDING
from datetime import datetime
modelp = joblib.load("qmodel.pkl")
currenttime = datetime.now()
client = MongoClient("mongodb+srv://moodtrackr:moodtrackr1@moodtrackr.hwwdhdl.mongodb.net/?retryWrites=true&w=majority&appName=AtlasApp")
db = client['test']
collection = db["modelpredictions"]
collection1=db["userfeelings"]

url_params = st.experimental_get_query_params()
user = url_params.get("user", [""])[0]

home_button_style = """
    background-color: #f63366;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    text-align: center;
    text-decoration: none;
    font-size: 16px;
    cursor: pointer;
    float: right;
"""

def navigate_to_home():
   
    home_url = "https://serenemindz-home.onrender.com/"
    st.markdown('<a href="{}" style="{}" target="_blank">Home</a>'.format(home_url, home_button_style), unsafe_allow_html=True)

navigate_to_home()

def predict_recent_data():
    recent_data = collection.find().sort([('_id', DESCENDING)]).limit(1)
    
    for item in recent_data:
        del item['_id']
        del item['__v']
        l1 = item.values()
        a = list(l1)
        print(a)
        a = np.array(a).reshape(1, -1)
        pred = modelp.predict(a)
        stress=pred[0,1]         #3
        anxiety=pred[0,0]
        data={'date':currenttime,'username':user,'stress':stress,'anxiety':anxiety}
        collection1.insert_one(data)
        return pred[0, 0], pred[0, 1]

st.title("Mood Prediction App")

if st.button("Analyse"):
    anxiety_level, stress_level = predict_recent_data()
    st.write("Anxiety Level:", anxiety_level)
    st.write("Stress Level:", stress_level)
    st.write("The results and insights provided by the Virtual Stress Assessment Tool are based on automated text analysis algorithms. They are intended for informational purposes only and should not be considered a substitute for professional mental health advice, diagnosis, or treatment.Always consult with a qualified healthcare provider or mental health professional for personalized guidance and support. We do not assume responsibility for any decisions made based on the results generated by this tool. Use this tool responsibly and seek professional assistance when needed.")



if st.button("View your previous status"):
    data = collection1.find({'username': user})
    for each in data:
        st.write("Visited on:", each["date"])
        st.write("Stress:", each["stress"])
        st.write("Anxiety:", each["anxiety"])
        st.write("\n")
