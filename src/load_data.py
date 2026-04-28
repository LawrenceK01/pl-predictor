import pandas as pd
from sqlalchemy import create_engine

RAW_CSV = "data/openpowerlifting-2026-04-25-e49ec9e6.csv"
DB_PATH = "data/powerlifting.db"

def load_to_sqlite():
    print("Reading CSV...")
    df = pd.read_csv(RAW_CSV, low_memory=False)

    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    print(df.dtypes)

    print("Writing to SQLite...")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df.to_sql("meets", engine, if_exists="replace", index=False)
    print(f"Done. Database saved to {DB_PATH}")

if __name__ == "__main__":
    load_to_sqlite()