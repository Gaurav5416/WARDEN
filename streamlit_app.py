import streamlit as st
import pandas as pd
from src.utils import mapped_spitter, get_reference_data
from pipelines.deployment_pipeline import deploying_warden
from steps.deployment.predictor import bentoml_predictor
from steps.data_reporter import data_reporter
from steps.deployment.prediction_service_loader import bentoml_prediction_service_loader
from configs.deployment_config import MODEL_NAME, PIPELINE_NAME, PIPELINE_STEP_NAME
import streamlit.components.v1 as components

st.title("Warden : credit card fraud detector")

st.markdown(
    """ 
    #### Problem Statement 
    Warden predicts whether the transaction is fraudulent or legitimate
    """
)

service = bentoml_prediction_service_loader(
        model_name=MODEL_NAME,
        pipeline_name=PIPELINE_NAME,
        step_name=PIPELINE_STEP_NAME,
        running=False,
    )

if service is None:
    st.write(
        "No service could be found. The pipeline will be run first to create a service."
        )
    deploying_warden()



def batch():
    st.write("Please upload your batch data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        prepared_df, filtered_df = bentoml_predictor(service=service, df=df)
        reference_data = get_reference_data()
        report_json, report_html = data_reporter(
            reference_dataset = reference_data, 
            comparison_dataset = prepared_df)
        
        if st.button("Predict") :
            st.write("Filtered DataFrame:")
            st.dataframe(filtered_df)

        st.download_button(label="Generate Report", data=report_html, file_name='Report.html')
        
        

        # batch(data)

def real_time():
    categories = ['entertainment', 'food_dining', 'gas_transport', 'grocery_net', 'grocery_pos', 'health_fitness', 'home', 'kids_pets', 'misc_net', 'misc_pos', 'personal_care', 'shopping_net', 'shopping_pos', 'travel']
    
    year = st.sidebar.radio("Select a year", [2019, 2020])
    day = st.sidebar.slider("Select a day", 1, 31)
    month = st.sidebar.slider("Select a month", 1, 12)
    weekday = st.sidebar.slider("Select a weekday", 1, 7)
    hour = st.sidebar.slider("Select hour", 0, 23)
    category = st.selectbox('Select a category', categories)
    mapped_category = mapped_spitter(category)
    amt = st.number_input("Amount", step=0.1, format="%.2f")
    gender = st.radio("Select Gender", ['Female', 'Male'])
    age = st.slider("Select age", 0, 100)
    mapped_gender = 1 if gender == 'Male' else 0

    df = pd.DataFrame({
        "category": [mapped_category],
        "amt": [amt],
        "gender_M": [mapped_gender],
        "trans_year": [year],
        "trans_month": [month],
        "trans_day": [day],
        "trans_hour": [hour],
        "trans_dayofweek": [weekday],
        "age": [age],
    })

    if st.button("Predict"):
        data = df.to_numpy()
        prediction = service.predict("predict_ndarray", data)

        if prediction == 1:
            st.warning("The transaction has a high chance of being fraudulent.")
        elif prediction == 0:
            st.success("The transaction is safe.")

# Display content based on selected tab

choice = st.sidebar.radio("Choose an option", ["Realtime inference", "Batch inference"])

if choice == "Realtime inference":
    real_time()
if choice == "Batch inference":
    batch()

