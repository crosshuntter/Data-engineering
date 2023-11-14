import os

import pandas as pd
# for geocoding
from opencage.geocoder import OpenCageGeocode
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
api_key = "e5370232998a4369b116891cd3297584"

from functions import *

# reading the csv file
green_taxi_df = pd.read_csv('../data/green_tripdata_2018-05.csv')
if(os.path.exists('../data/cleaned_data.csv')):
    green_taxi_df_clean = pd.read_csv('../data/cleaned_data.csv')
else:
    green_taxi_df_clean = green_taxi_df.copy()
    ## 1 - Data cleaning
    ### Renaming columns
    rename_columns(green_taxi_df_clean)

    ### Observing and handling inconsistent and incorrect data
    green_taxi_df_clean = handle_inconsistent_incorrect(green_taxi_df_clean)

    ### add borough and zone columns to DataFrame
    green_taxi_df_clean = extract_location_info(green_taxi_df_clean)
    ### Observing and handling Missing Data and placeholders
    green_taxi_df_clean = handle_missing_unknown(green_taxi_df_clean)

    ### Observing outliers
    ## passenger_count
    green_taxi_df_clean = handle_passenger_count_outliers(green_taxi_df_clean)

    ## Trip_distance
    green_taxi_df_clean = handle_trip_distance_outliers(green_taxi_df_clean)

    ## fare_amount
    green_taxi_df_clean = handle_fare_amount_outliers(green_taxi_df_clean)

    ## tip_amoun
    green_taxi_df_clean = handle_tip_amount_iqr(green_taxi_df_clean)

    ## tolls_amount
    green_taxi_df_clean = handle_tolls_amount(green_taxi_df_clean)

    ## total_amount
    green_taxi_df_clean = handle_total_amount_outliers(green_taxi_df_clean)

    # 4 - Data transformation and feature eng.
    ## 4.1 - Discretization
    green_taxi_df_clean = generate_date_features(green_taxi_df_clean)

    ## 4.2 - Adding more features(feature eng.)
    green_taxi_df_clean = generate_time_features(green_taxi_df_clean)

    ## 4.3 - Encoding
    green_taxi_df_clean = encode_features(green_taxi_df_clean)

    ## 4.4 - Normalisation 
    # green_taxi_df_clean = scale_features(green_taxi_df_clean)

    ## 4.5 - Additional data extraction (GPS coordinates)
    # green_taxi_df_clean = add_coordinates_to_dataframe(green_taxi_df_clean)

    ## 5- Exporting the dataframe to a csv file or parquet
    green_taxi_df_clean.to_csv('../data/cleaned_data.csv',index=False)

engine = create_engine('postgresql://root:root@pgdatabase:5432/green_taxi')

if(engine.connect()):
	print('connected succesfully')
else:
	print('failed to connect')

try:
    green_taxi_df_clean.to_sql(name='green_taxi_5_2018', con=engine, if_exists='fail', index=False)
except IntegrityError:
    print("Table 'green_taxi_5_2018' already exists in the database.")
except Exception as e:
      print(f"An error occurred: {e}")