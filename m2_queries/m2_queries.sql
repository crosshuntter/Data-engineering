--query 1
SELECT *
FROM green_taxi_5_2018
ORDER BY trip_distance DESC
LIMIT 20;

--query 2
SELECT payment_type, AVG(fare_amount) AS avg_fare_amount
FROM green_taxi_5_2018
GROUP BY payment_type;

--query 3
SELECT pu_location_borough, AVG(tip_amount) AS avg_tip_amount
FROM green_taxi_5_2018
GROUP BY pu_location_borough
ORDER BY avg_tip_amount DESC
LIMIT 1;

--query 4

SELECT pu_location_borough, AVG(tip_amount) AS avg_tip_amount
FROM green_taxi_5_2018
GROUP BY pu_location_borough
ORDER BY avg_tip_amount 
LIMIT 1;

--query 5
SELECT do_location, COUNT(*) AS trip_count
FROM green_taxi_5_2018
WHERE is_weekend = true
GROUP BY do_location
ORDER BY trip_count DESC
LIMIT 1;

--query 6
SELECT trip_type, AVG(trip_distance) AS avg_trip_distance
FROM green_taxi_5_2018
GROUP BY trip_type;

--query 7
SELECT AVG(fare_amount) AS avg_fare_amount
FROM green_taxi_5_2018
WHERE EXTRACT(HOUR FROM CAST(lpep_pickup_datetime AS TIMESTAMP)) BETWEEN 16 AND 18;