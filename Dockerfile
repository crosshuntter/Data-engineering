FROM python:3.11

RUN pip install pandas scikit-learn numpy scipy

RUN pip install opencage

RUN pip install sqlalchemy psycopg2



ENTRYPOINT ["python","src/main.py"]