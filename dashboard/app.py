import streamlit as st
import pandas as pd
import requests
import os

st.title("Fraud Detection Dashboard")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    
    if st.button("Run Batch Prediction"):
        records = df.to_dict(orient="records")
        res = requests.post("http://localhost:8000/predict/batch", json=records)
        if res.status_code == 200:
            preds = res.json()["predictions"]
            df_res = pd.DataFrame(preds)
            st.write(df_res)
            st.bar_chart(df_res["probability"])
        else:
            st.error("Prediction failed")
            
    if st.button("Check Drift"):
        records = df.to_dict(orient="records")
        res = requests.post("http://localhost:8000/drift", json=records)
        if res.status_code == 200:
            report = res.json()["report"]
            st.json(report)
        else:
            st.error("Drift check failed")
            
if os.path.exists("reports/shap_summary.png"):
    st.image("reports/shap_summary.png", caption="SHAP Summary")
