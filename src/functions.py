import pandas as pd
from scipy import stats
import numpy as np
# For min_max scaling
from sklearn.preprocessing import MinMaxScaler

# For Label Encoding
from sklearn import preprocessing

from opencage.geocoder import OpenCageGeocode
api_key = "e5370232998a4369b116891cd3297584"


def rename_columns(df):
#     make all cols lower case
    df.columns = df.columns.str.lower()

    df.columns = [col.replace(' ', '_') for col in df.columns]


def handle_inconsistent_incorrect(df):
    # Drop duplicate rows based on specified columns
    df = df.drop_duplicates(subset=['lpep_pickup_datetime', 'lpep_dropoff_datetime', 'pu_location', 'do_location', 'trip_distance'], keep='last')

    # Convert datetime columns to datetime objects
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])
    

    # Keep only trips that happened in 2018 and in May
    df = df[(df['lpep_pickup_datetime'].dt.year == 2018) & (df['lpep_dropoff_datetime'].dt.year == 2018)]
    df = df[df['lpep_pickup_datetime'].dt.month == 5]

    # Keep only trips with positive distance, fare, and correct time order
    df = df[(df['fare_amount'] > 0) & (df['trip_distance'] > 0) & (df['lpep_pickup_datetime'] < df['lpep_dropoff_datetime'])]


    return df

def extract_location_info(df):
    # Split PU Location and DO Location into borough and zone
    df["pu_location_borough"] = df["pu_location"].str.split(',').str[0]
    df["pu_location_zone"] = df["pu_location"].str.split(',').str[1]
    df["do_location_borough"] = df["do_location"].str.split(',').str[0]
    df["do_location_zone"] = df["do_location"].str.split(',').str[1]
    return df

def handle_missing_unknown(df):
    # Drop specified columns
    df = df.drop(['ehail_fee', 'congestion_surcharge'], axis=1)

    # Filter out rows with unknown boroughs
    df = df[(df["pu_location_borough"] != "Unknown") & (df["do_location_borough"] != "Unknown")]

    # Replace unknown payment types based on tip amount
    for index, row in df.iterrows():
        if pd.isnull(row['payment_type']) or row['payment_type'] == 'Uknown':
            if row['tip_amount'] == 0:
                df.at[index, 'payment_type'] = 'Cash'
            else:
                df.at[index, 'payment_type'] = 'Credit card'

    # Handling missing data in 'extra' column
    df['extra_imp'] = df['extra'].fillna(0)

    return df

def handle_passenger_count_outliers(df):
    df = df[df.passenger_count <= 6]
    df["passenger_count"] = df["passenger_count"].astype(int)
    return df

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

def handle_fare_amount_outliers(df):
    # Calculate Z-scores for the specified column
    z = np.abs(stats.zscore(df['fare_amount']))
    filtered_entries = z < 3.5
    
    # Impute outliers with the maximum value within Z-score threshold
    max_within_threshold = df[filtered_entries]["fare_amount"].max()
    df["fare_amount_imputed"] = df["fare_amount"]
    df.loc[~filtered_entries, "fare_amount_imputed"] = max_within_threshold
    
    return df

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

def handle_tolls_amount(df):
    # Calculate the IQR (Interquartile Range)
    
    # Define lower and upper bounds for outliers
    
    # Identify outliers based on IQR bounds
    outliers_mask = (df["tolls_amount"] > 5.76)
    
    # Impute outliers with the upper limit of the range containing majority of values
    upper_limit = 5.76
    df.loc[outliers_mask, "tolls_amount"] = upper_limit
    
    return df

def handle_total_amount_outliers(df):
    # Calculate Z-scores for the specified column
    z = np.abs(stats.zscore(df['total_amount']))
    filtered_entries = z < 3.5
    
    # Impute outliers with the maximum value within Z-score threshold
    max_within_threshold = df[filtered_entries]["total_amount"].max()
    df.loc[~filtered_entries, "total_amount"] = max_within_threshold
    
    return df

def generate_date_features(df):
    # Create 'week_number' column
    df['week_number'] = df["lpep_pickup_datetime"].dt.isocalendar().week
    
    # Create 'date_range' column
    df['date_range'] = df["lpep_pickup_datetime"].dt.to_period('W')
    
    return df

def generate_time_features(df):
    # Create 'is_weekend' column (1 for weekend, 0 for weekday)
    df['is_weekend'] = df["lpep_pickup_datetime"].dt.dayofweek // 5 == 1
    
    # Create 'is_night' column (1 for night, 0 for day)
    df['is_night'] = (df["lpep_pickup_datetime"].dt.hour < 6) | (df["lpep_pickup_datetime"].dt.hour >= 20)
    
    return df

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

def scale_features(df):
    result = df.copy() # take a copy of the dataframe
    columns_to_scale = ["trip_distance","fare_amount","tip_amount","total_amount"]
    # apply min-max scaling to the specified columns
    scaler = MinMaxScaler()
    result[columns_to_scale] = scaler.fit_transform(result[columns_to_scale])
    
    return result

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