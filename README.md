# 🛡️ WARDEN : Credit Card Fraud Detector 
Presenting ✨ the almighty **WARDEN** 🛡️. Warden is a state-of-the-art End to End MLOps project designed for credit card fraud detection. Leveraging advance machine learning algorithms, MLOps best practices. Warden incorporates Alerting and monitoring system with robust model development and deployment. 

With its intuitive Streamlit interface, Warden offers both batch and real-time inference, accompanied by comprehensive reports. Built-in monitoring guarantees continuous data quality assessment, making Warden a formidable guardian against fraudulent transactions.

This project aims to streamline the process of filtering the fraud transactions, providing insights, comprehensive reports and predictions to drive impactful business decisions! 📊✨


# Table of Contents 📑
+ [Problem Statement](#problem-statement-) ❓
+ [Solution Approach](#solution-approach-) 🎯
+ [Folder Structure](#folder-structure-) 📁
+ [Project Overview](#project-overview-) 👁️
+ [Prerequisites](#prerequisites-) 📋
+ [Running the project](#running-the-project-) 🏃‍♂️


# Problem Statement ❓
In 2022, U.S. credit card transactions totaled 54.8 billion for an average of 150.15 million per day, 6.25 million per hour, 104,274 per minute. or 1,739 per second. 
The firm estimated the global credit card transaction to be 21,510 per second. [link to the report ](https://capitaloneshopping.com/research/number-of-credit-card-transactions/)

As the information suggests 🤔 , It is  impossible for a human to check each and every transaction let alone detect  🔍 which one is fraud or legitimate . . . 



# Solution Approach 🎯
1. Researchers have found some common trends and patterns when a transaction is fraud suggesting the solution to be **Statistical Analysis**

2. But the amount of data still daunts us, 21k+ transactions per second is a mammoth data and can't be handled and analysed by statisticians alone. we need machine to do the work for us which leads us to **Artificial intelligence**


Upon reviewing the statements, its clear that the solution of the problem lies in **Machine learning** which leverages both statistical algorithms and artificial intelligence reducing human work load, cost and time.


The project proposes : an **E2E(End-to-End) ML solution** in which User can input details of transactions and recieve insight about the transaction whether its fraud or legit based on trends and patterns.



# Project Overview 👁️

Our project consists of several stages:

## Data Collection 📁
Due to its sensitive nature, Real world Credit card transaction data is often difficult to get hands on due to monetary and confidentiality issues, therefore we have used a [kaggle dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) which is "synthetically generated".

## Data Handling 💾
We used ORM (Object Relational Mapping) to handle the data. We shifted the data to a Postgresql server and fetched it from the database using ORM. Please find ORM steps and code in `data/management` folder.

## Exploratory Data Analysis 🔍
In EDA, we perfomed initial exploration of the data, familiarity with the features, Inspected the nature of the data through visualizations and various other statistical techniques such as Cross validation. We also noted down the necessary data pipeline steps we need to take for E2E pipeline. Please make sure to check our comprehensive [EDA Notebook](./notebooks/EDA.ipynb)

## ETL Data Pipeline 🚚
After initial exploration and identifying features that needed to be dealt with. We moved towards ETL in which we developed a ETL data pipeline involving necessary encodings and engineering to features and balancing the data through which we made the data into a format suitable for machine learning. Please find the necessary ETL steps in `steps/ETL` folder.

## Data Testing 📝
Data testing is a part of monitoring system in WARDEN. Before passing the data to training, the data passes through data testing phase which conducts series of tests including : integrity checks, missing value handling, outlier detection, and feature distribution analysis. These tests can be aquired through dashboard anytime and are also alerted through slack. Please find the data testing code in `steps/monitoring/data_tester.py` file.

## Slack Alerts 🚨
After the testing phase, the results are alerted to team through slackbot. Streamlining the process of alerting and monitoring. Please find the alerting code in `steps/monitoring/alerter.py` and `src/utils.py` files.

## Model Building 🏗️
Through EDA, after testing through various algorithms we discovered that "Adaboost" performed better than other algorithm, therefore after data testing we trained our model on Adaptive boosting.

## Model Evaluation 📊
Since its a classification problem and a problem in which False Negatives (Type II error) are more expensive than False Positives (Type I error) we paid special focus on Recall, ROC-AUC scores and F1 Score. Please find training and evaluation steps in `steps/training` folder.

## Model Deployment 🚀
We deployed model using BentoML and create a user-friendly web application using Streamlit. Users can input transaction details, and the model predicts wether the transaction is fraudualent or not. This application might help for pre-screening,enabling early identification of potential fraud transactions filtering them for further analysis. Please find the deployment steps in `steps/deployment` folder.

## Reports 📑
Reports is another tool of Monitoring system. Whenever warden recieves a batch inference request,it auto-generates reports consisting of model performance decay, data drift, target drift, and many other useful metrics with beautiful visualizations. Please find the report code in `steps/monitoring/data_reporter.py` file.



# Folder structure 📁

```
├── assets 📁
│   └── ...
├── configs 📁
│   ├── deployment_config.py
│   └── etl_config.py
├── data 📁
│   ├── fraud.csv
│   ├── sample_batch_data.csv
│   └── management 📁
│       ├── fill_table.py
│       ├── index.py
│       └── retriever.py
├── materializer 📁
│   └── custom_materializer.py
├── notebooks 📁
│   └── EDA.ipynb
├── pipelines 📁
│   ├── deployment_pipeline.py
│   ├── inference_pipeline.py
│   └── training_pipeline.py
├── src 📁
│   ├── etl 📁
│   │   ├── balance_data.py
│   │   ├── categorical_encoding.py
│   │   ├── data_ingestion.py
│   │   └── feature_engineering.py
│   ├── training 📁
│   │   ├── model_building.py
│   │   └── model_evaluation.py
│   └── utils.py
├── steps 📁
│   ├── monitoring 📁
│   │   ├── alerter.py
│   │   ├── data_reporter.py
│   │   └── data_tester.py
│   ├── deployment 📁
│   │   ├── bento_builder.py
│   │   ├── deployment_trigger.py
│   │   ├── dynamic_importer.py
│   │   ├── model_loader.py
│   │   ├── prediction_service_loader.py
│   │   └── predictor.py
│   ├── etl 📁
│   │   ├── categorical_encoder.py
│   │   ├── data_balancer.py
│   │   ├── data_loader.py
│   │   ├── data_splitter.py
│   │   └── feature_engineer.py
│   └── training 📁
│       ├── evaluate_model.py
│       └── train_model.py
├── service.py
├── streamlit_app.py
├── requirements.txt
├── README.md
└── run.py

```

# Features 🎨
The important features and tech stack used in this project are as follows :-

## ORM (Object Relational Mapping) 
 Object Relational Mapping (ORM) simplifies database interactions by allowing us to work with database entities as Python objects. In this project, We used ORM to push data to and fetch data from a PostgreSQL server, abstracting away the need for direct SQL queries and making database operations more intuitive and manageable. ORM steps and code is in `data/management` folder.
 
 - After Running `data/management/index.py`
![ORM Table Creation](assets/ORM_table_creation.png)

 - After Running `data/management/fill_table.py`
![ORM Table Filled](assets/ORM_table_filled.png)


## ZenML
A comprehensive data science project often requires dealing with various industry tools and technologies for experiment tracking, testing, alerting, monitoring, deployment. These tools are useful but difficult to use without a centralized framework. 


ZenML is open source **MLOps framework** to 
streamline and automate the end-to-end machine learning lifecycle. Its flexible architecture allows for easy integration with popular tools and frameworks. ZenML provides a dashboard to keep track of pipelines, runs, artifacts, stack components and many more . . . 

- **Here is a glimps of ZenML dashboard**
![ZenML Dashboard](assets/zenml_runs.png)

- **ZenML also provides a beautiful DAGs to understand the flow**
![ZenML Dag](assets/dag.png)


- **The various popular tools integrated and streamlined with ZenML are here.**
![ZenML Dashboard](assets/zenml_stack.png)

## BentoML
To Deploy the project we used **BentoML Model Deployer** component. BentoML creates a self sufficient Bento with all necessary components in a unified Bento. This bento then can be deployed anywhere on cloud for predictions.

![BentoML deployer](assets/deploy.png)

## MLFlow
Every run in ML cycle is an experiment. We used MLFlow for **experiment tracking** in our project. MLFlow comes with its intuitive dashboard to track experiments, logs parameters, metrics and many more ...

![MLflow Dashboard](assets/mlflow_dashboard.png)


## EvidentlyAI
EvidentlyAI is a significant component in our project's **Model Monitoring system**. 

It helps in monitoring the performance and behavior of machine learning models in production, providing insights into model drift, data drift, and other relevant metrics for effective monitoring and management of deployed models.

We used Evidently for 2 purposes :
- Generating a comprehensive report after batch inference for various evaluations like model drift, data drift, classification performance with beautiful and interactive visualizations. Here is a [Sample Report](https://batch-report.netlify.app)

- Conducting tests on data before training the model on it. These tests are then alerted to slack channel through slackbot. Here is the [Sample Test Suite](https://test-suite.netlify.app)

## Slackbot Alerts
We integrated slackbot in our pipeline so that the tests results are notified to our team directly through slackbot in our channel. Slack alerters is another major part of our Alerting and Monitoring System.

![Slack Alerts](assets/slackbot.png)

# Running the project 🏃‍♂️

### Prerequisites 📋
Before starting, ensure you have the following prerequisites:
- **Python 3.10**
- **Ubuntu**
- **VS Code (WSL)**
- **PostgreSQL server**
- **pgadmin4**

Please follow following steps to run the project on your machine locally . . . 

1. Clone this repository to your local machine.

    ```
    git clone https://github.com/Gaurav5416/WARDEN
    cd WARDEN
    ```

2. Install necessary requirements.
    ```
    pip install -r requirements.txt
    ```
3. Wake up zenml
    ```
    zenml init
    zenml up
    ```
4. Installing zenml integrations
    ```
    zenml integration install evidently -y
    zenml integration install bentoml -y
    zenml integration install mlflow -y
    zenml integration install slack -y
    ```

5. Registering necessary stack components
    ```
    zenml data-validator register evidently_validator --flavor=evidently
    zenml model-deployer register bentoml_deployer --flavor=bentoml
    zenml experiment-tracker register mlflow_tracker --flavor=mlflow
    zenml alerter register slack_alerter --flavor=slack --slack_token=<> --default_slack_channel_id=<>
    zenml stack register warden_stack -d bentoml_deployer -e mlflow_tracker -al slack_alerter -dv evidently_validator -o default -a default
    zenml stack set warden_stack
    ```

6. Create a .env file specifying datapath of table and database path eg..
    ```
    DB_URL = "postgresql://username:password@host:port/database"
    datapath = "WARDEN/data/fraud.csv"
    ```

7. Create SQL table and push CSV data into postgreSQL database using ORM
    ```
    cd data/management
    python index.py
    python fill_table.py
    ```

8. Deploying the ML Model
    ```
    python run.py
    ```

9. Running Streamlit app
    ```
    streamlit run streamlit_app.py
    ```

***