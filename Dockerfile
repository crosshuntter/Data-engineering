FROM python:3.11

RUN pip install pandas

RUN pip install sqlalchemy psycopg2

WORKDIR /app

copy src/ src/

copy data/ data/

ENTRYPOINT ["python","src/test.py"]