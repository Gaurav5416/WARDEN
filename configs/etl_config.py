table_name = "fraud" # Passed to ingest_data
target_col = "is_fraud" #Passed to data_splitter balance _data
cols_to_drop = [ 
                    # 'id', 
                    'cc_num', 
                    'merchant', 
                    'first', 
                    'last', 'city', 
                    'state', 
                    'street', 
                    'zip', 'lat', 
                    'long', 
                    'job', 
                    'trans_num', 
                    'city_pop', 
                    'unix_time', 
                    'merch_lat', 
                    'merch_long',
                ] # Passed to feature engineering
mean_encoding_columns = ['category'] # Passed to categorical encoding
binary_encoding_columns = ['gender'] # Passed to categorical encoding
standard_scaling_columns = ['amt'] # Passed to feature engineering
feature_engineering_columns = ["trans_date_trans_time", "dob"] # Passed to feature engineering