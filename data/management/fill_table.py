import pandas as pd
from index import Fraud, Session
import os
from dotenv import load_dotenv

load_dotenv()

datapath = os.getenv("datapath")

batch_size = 20000
data_reader = pd.read_csv(datapath, chunksize=batch_size)

with Session.begin() as db:
    for chunk in data_reader:
        chunk.drop(columns=['Unnamed: 0'], inplace=True)
        # Convert DataFrame rows into dictionaries
        fraud_data = chunk.to_dict(orient='records')
        # Bulk insert the data into the database
        db.bulk_save_objects([Fraud(**row) for row in fraud_data])
