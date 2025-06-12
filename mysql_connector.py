from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

# Setup SQLAlchemy engine
engine = create_engine("mysql+pymysql://root:root@localhost/retail_data", connect_args={"connect_timeout": 5})

# Query the database
try:
    with engine.connect() as conn:
        query = text("SELECT * FROM cleaned_transactions LIMIT 100;")
        df = pd.read_sql(query, conn)
        print(df.head())
except SQLAlchemyError as e:
    print(f"❌ Error connecting to database: {e}")
