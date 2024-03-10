import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from steps.deployment.prediction_service_loader import bentoml_prediction_service_loader
from pipelines.deployment_pipeline import deploying_warden
from configs.deployment_config import MODEL_NAME, PIPELINE_NAME, PIPELINE_STEP_NAME
from src.utils import mapped_spitter

def main():
    st.title("Warden : credit card fraud detector")
    st.markdown(
        """ 
    #### Problem Statement 
     Warden predicts whether the transaction is fraudelent or legitimate   """
    )

    categories = [
        'entertainment',
        'food_dining',
        'gas_transport',
        'grocery_net',
        'grocery_pos',
        'health_fitness',
        'home',
        'kids_pets',
        'misc_net',
        'misc_pos',
        'personal_care',
        'shopping_net',
        'shopping_pos',
        'travel'
        ]
    min_date = datetime(2019, 1, 1)
    max_date = datetime(2020, 12, 31)

    date = st.date_input("Select transaction date", min_value=min_date, max_value=max_date, value=min_date)
    day = date.day
    month = date.month
    year = date.year
    weekday = date.weekday()

    hour = st.slider("Select hour", 0, 23)

    category = st.selectbox('Select a category ?', categories)
    mapped_category = mapped_spitter(category)

    amt = st.number_input("Amount", step=0.1, format="%.2f")
    
    gender = st.sidebar.radio("Select Gender", ['Female', 'Male'])
    mapped_gender = 1 if gender == 'Male' else 0




    age = st.slider("Select age", 0, 100)

    df = pd.DataFrame(
        {
            "category": [mapped_category],
            "amt": [amt],
            "gender_M": [mapped_gender],
            "trans_year": [year],
            "trans_month": [month],
            "trans_day": [day],
            "trans_hour": [hour],
            "trans_dayofweek": [weekday],
            "age": [age],
        }
    )
    # df = df.round(2)
    # print(df.dtypes)
    # df = df.astype(float)
    # print(df.head())

    if st.button("Predict"):
        service = bentoml_prediction_service_loader(
         model_name =  MODEL_NAME,
         pipeline_name= PIPELINE_NAME,
         step_name= PIPELINE_STEP_NAME,
         running = False,
         )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            deploying_warden()
            
        data = df.to_numpy()
        prediction = service.predict("predict_ndarray", data)
        
        if prediction == 1 :
            # st.success("The transaction has high chances of being fraud")
            st.warning("The transaction has high chances of being fraud")

        if prediction == 0 :
            st.success("The transaction is safe")
            

if __name__ == "__main__":
    main()
