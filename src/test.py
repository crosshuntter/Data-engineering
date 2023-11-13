import pandas as pd

from sqlalchemy import create_engine

df = pd.read_csv('/app/data/green_tripdata_2018-05.csv')

engine = create_engine('postgresql://root:root@pgdatabase:5432/titanic')

if(engine.connect()):
	print('connected succesfully')
else:
	print('failed to connect')


df.to_sql(name = 'green_taxi_dataset',con = engine,if_exists='replace')

