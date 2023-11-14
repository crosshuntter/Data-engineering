import pandas as pd
from scipy import stats
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
# for geocoding
from opencage.geocoder import OpenCageGeocode
api_key = "e5370232998a4369b116891cd3297584"

from functions import *

# reading the csv file
green_taxi_df = pd.read_csv('../data/green_tripdata_2018-05.csv')
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
def handle_passenger_count_outliers(df):
    df = df[df.passenger_count <= 6]
    return df
green_taxi_df_clean = handle_passenger_count_outliers(green_taxi_df_clean)
## Trip_distance
def handle_trip_distance_outliers(df):
    # Calculate Z-scores for the specified column
    df = df[df['trip_distance'] < 100]
    z = np.abs(stats.zscore(df['trip_distance']))
    filtered_entries = z < 3.5
    
    # Impute outliers with the maximum value within Z-score threshold
    max_within_threshold = df[filtered_entries]["trip_distance"].max()
    df["trip_distance_imputed"] = df["trip_distance"]
    df.loc[~filtered_entries, "trip_distance_imputed"] = max_within_threshold
    
    return df
green_taxi_df_clean = handle_trip_distance_outliers(green_taxi_df_clean)

## fare_amount
def handle_fare_amount_outliers(df):
    # Calculate Z-scores for the specified column
    z = np.abs(stats.zscore(df['fare_amount']))
    filtered_entries = z < 3.5
    
    # Impute outliers with the maximum value within Z-score threshold
    max_within_threshold = df[filtered_entries]["fare_amount"].max()
    df["fare_amount_imputed"] = df["fare_amount"]
    df.loc[~filtered_entries, "fare_amount_imputed"] = max_within_threshold
    
    return df
green_taxi_df_clean = handle_fare_amount_outliers(green_taxi_df_clean)
## tip_amount
def handle_tip_amount_iqr(df):
    # Calculate the IQR (Interquartile Range)
    Q1 = df["tip_amount"].quantile(0.25)
    Q3 = df["tip_amount"].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define lower and upper bounds for outliers
    upper_bound = Q3 + 3 * IQR
    
    # Identify outliers based on IQR bounds
    outliers_mask = (df["tip_amount"] > upper_bound)
    
    # Impute outliers with the upper limit of the IQR
    upper_limit = Q3 + 3 * IQR
    df.loc[outliers_mask, "tip_amount"] = upper_limit
    
    return df
green_taxi_df_clean = handle_tip_amount_iqr(green_taxi_df_clean)

## tolls_amount
def handle_tolls_amount(df):
    # Calculate the IQR (Interquartile Range)
    
    # Define lower and upper bounds for outliers
    
    # Identify outliers based on IQR bounds
    outliers_mask = (df["tolls_amount"] > 5.76)
    
    # Impute outliers with the upper limit of the range containing majority of values
    upper_limit = 5.76
    df.loc[outliers_mask, "tolls_amount"] = upper_limit
    
    return df

green_taxi_df_clean = handle_tolls_amount(green_taxi_df_clean)

## total_amount
def handle_total_amount_outliers(df):
    # Calculate Z-scores for the specified column
    z = np.abs(stats.zscore(df['total_amount']))
    filtered_entries = z < 3.5
    
    # Impute outliers with the maximum value within Z-score threshold
    max_within_threshold = df[filtered_entries]["total_amount"].max()
    df.loc[~filtered_entries, "total_amount"] = max_within_threshold
    
    return df
green_taxi_df_clean = handle_total_amount_outliers(green_taxi_df_clean)

# 4 - Data transformation and feature eng.
## 4.1 - Discretization
def generate_date_features(df):
    # Create 'week_number' column
    df['week_number'] = df["lpep_pickup_datetime"].dt.isocalendar().week
    
    # Create 'date_range' column
    df['date_range'] = df["lpep_pickup_datetime"].dt.to_period('W')
    
    return df


green_taxi_df_clean = generate_date_features(green_taxi_df_clean)

## 4.2 - Adding more features(feature eng.)
def generate_time_features(df):
    # Create 'is_weekend' column (1 for weekend, 0 for weekday)
    df['is_weekend'] = df["lpep_pickup_datetime"].dt.dayofweek // 5 == 1
    
    # Create 'is_night' column (1 for night, 0 for day)
    df['is_night'] = (df["lpep_pickup_datetime"].dt.hour < 6) | (df["lpep_pickup_datetime"].dt.hour >= 20)
    
    return df
green_taxi_df_clean = generate_time_features(green_taxi_df_clean)



## 4.3 - Encoding

def encode_features(df):
    result = df.copy() # take a copy of the dataframe
    label_encoding_columns = ['store_and_fwd_flag']
    one_hot_encoding_columns = ['vendor',"rate_type", "payment_type", 'trip_type']
    # apply label encoding to the specified columns
    label_encoder = preprocessing.LabelEncoder()
    for col in label_encoding_columns:
        result[col] = label_encoder.fit_transform(result[col])

    # apply one-hot encoding to the specified columns using get_dummies
    one_hot_encoding_df = pd.get_dummies(result[one_hot_encoding_columns])
    
    result = pd.concat([result, one_hot_encoding_df], axis=1)
    

    return result

green_taxi_df_clean = encode_features(green_taxi_df_clean)

## 4.4 - Normalisation 
def scale_features(df):
    result = df.copy() # take a copy of the dataframe
    columns_to_scale = ["trip_distance","fare_amount","tip_amount","total_amount"]
    # apply min-max scaling to the specified columns
    scaler = MinMaxScaler()
    result[columns_to_scale] = scaler.fit_transform(result[columns_to_scale])
    
    return result

green_taxi_df_clean = scale_features(green_taxi_df_clean)
## 4.5 - Additional data extraction (GPS coordinates)


def get_unique_coordinates(unique_locations):
    geocoder = OpenCageGeocode(api_key)
    coordinates_dict = {}

    for location in unique_locations:
        results = geocoder.geocode(location)
        if results and len(results):
            coordinates_dict[location] = {
                'latitude': results[0]['geometry']['lat'],
                'longitude': results[0]['geometry']['lng']
            }

    return coordinates_dict


def add_coordinates_to_dataframe(df):
    unique_pickup_locations = df["pu_location"].unique()
    unique_dropoff_locations = df["do_location"].unique()
    unique_locations = np.unique(np.concatenate((unique_pickup_locations, unique_dropoff_locations), axis=0))


    unique_coordinates = get_unique_coordinates(unique_locations)
    # dropoff_coordinates = get_unique_coordinates(unique_dropoff_locations)

    df['pu_latitude'] = df["pu_location"].map(lambda x: unique_coordinates.get(x, {}).get('latitude'))
    df['pu_longitude'] = df["pu_location"].map(lambda x: unique_coordinates.get(x, {}).get('longitude'))
    df['do_latitude'] = df["do_location"].map(lambda x: unique_coordinates.get(x, {}).get('latitude'))
    df['do_longitude'] = df["do_location"].map(lambda x: unique_coordinates.get(x, {}).get('longitude'))

    return df

green_taxi_df_clean = add_coordinates_to_dataframe(green_taxi_df_clean)


## 5- Exporting the dataframe to a csv file or parquet
green_taxi_df_clean.to_csv('../dataset/cleaned_data.csv',index=False)