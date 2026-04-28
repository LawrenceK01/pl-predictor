import os
import pandas as pd
from sqlalchemy import create_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "powerlifting.db")

def load_clean_data():
    engine = create_engine(f"sqlite:///{DB_PATH}")

    query = """
        SELECT
            Sex,
            Age,
            BodyweightKg,
            TotalKg,
            Equipment,
            Event
        FROM meets
        WHERE Equipment = 'Raw'
            AND Event = 'SBD'
            AND TotalKg IS NOT NULL
            AND Age IS NOT NULL
            AND BodyweightKg IS NOT NULL
            AND Sex IN ('M', 'F')
            AND Place NOT IN ('DQ', 'G', 'NS')
            AND Age >= 14
            AND Age <= 80
        """
    df = pd.read_sql(query, engine)
    print(f"Clean dataset: {len(df):,} rows")
    return df

if __name__ == "__main__":
    df = load_clean_data()
    print(df.describe())