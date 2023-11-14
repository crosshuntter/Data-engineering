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
    df["passenger_count"] = df["passenger_count"].astype(int)

    # Keep only trips that happened in 2018 and in May
    df = df[(df['lpep_pickup_datetime'].dt.year == 2018) & (df['lpep_dropoff_datetime'].dt.year == 2018)]
    df = df[df['lpep_pickup_datetime'].dt.month == 5]

    # Keep only trips with positive distance, fare, and correct time order
    df = df[(df['fare_amount'] > 0) & (df['trip_distance'] > 0) & (df['lpep_pickup_datetime'] < df['lpep_dropoff_datetime'])]

    # Keep only trips with passenger count less than or equal to 10
    df = df[df['passenger_count'] <= 10]

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