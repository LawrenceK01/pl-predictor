import os
import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "powerlifting.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "progression_model.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "progression_encoder.joblib")

MAX_MEETS_PER_LIFTER = 15

def load_raw_data():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    query = """
        SELECT Name, Sex, Age, BodyweightKg, TotalKg, Goodlift, Date
        FROM meets
        WHERE Equipment = 'Raw'
          AND Event = 'SBD'
          AND TotalKg IS NOT NULL
          AND Age IS NOT NULL
          AND BodyweightKg IS NOT NULL
          AND Goodlift IS NOT NULL
          AND Sex IN ('M', 'F')
          AND Place NOT IN ('DQ', 'G', 'NS')
          AND Age >= 14
          AND Age <= 80
        ORDER BY Name, Date
    """
    df = pd.read_sql(query, engine)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def engineer_features(df):
    """
    For each lifter with 2+ meets, create 1 row per meet transition.
    Each row represents: given everything about meet N, predict total at meet N+1.
    """

    rows = []
    for name, group in df.groupby('Name'):
        group = group.sort_values('Date').reset_index(drop=True)

        # Cap at MAX_MEETS_PER_LIFTER
        group = group.head(MAX_MEETS_PER_LIFTER)

        if len(group) < 2:
            continue

        for i in range(len(group) - 1):
            current = group.iloc[i]
            next_meet = group.iloc[i + 1]

            months_between = (next_meet['Date'] - current['Date']).days / 30.44

            # Skip if time between meets is unrealistically long (5+ years) or negative
            if months_between <= 0 or months_between > 60:
                continue

            # Skip if bw change is unrealistically large (50+ kg)
            bw_change = next_meet['BodyweightKg'] - current['BodyweightKg']
            if abs(bw_change) > 50:
                continue

            rows.append({
                'Sex':              current['Sex'],
                'prev_total':       current['TotalKg'],
                'prev_gl':          current['Goodlift'],
                'prev_bodyweight':  current['BodyweightKg'],
                'next_bodyweight':  next_meet['BodyweightKg'],
                'bw_change':        bw_change,
                'age_at_next':      next_meet['Age'],
                'months_between':   months_between,
                'meet_number':      i + 1,  # which meet in lifter's career
                'next_total':       next_meet['TotalKg'],
            })

    return pd.DataFrame(rows)
    
def train():
    print("Loading raw data...")
    df = load_raw_data()
    print(f"Raw rows: {len(df):,}")

    print("Engineering features...")
    transitions = engineer_features(df)
    print(f"Transition rows: {len(transitions):,}")
    print(transitions.describe())

    # Encode sex
    le = LabelEncoder()
    transitions['Sex'] = le.fit_transform(transitions['Sex'])

    features = [
        'Sex', 'prev_total', 'prev_gl', 'prev_bodyweight', 'next_bodyweight',
        'bw_change', 'age_at_next', 'months_between', 'meet_number'
    ]

    X = transitions[features]
    y = transitions['next_total']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"MAE: {mae:.2f} kg")
    print(f"R2: {r2:.4f}")

    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"Model saved")

    return model, le, X_test, y_test, preds, transitions

if __name__ == "__main__":
    train()

