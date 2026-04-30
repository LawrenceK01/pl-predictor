import os
import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'powerlifting.db')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'full_history_model.joblib')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'full_history_encoder.joblib')

MAX_MEETS_PER_LIFTER = 20

def load_raw_data():
    engine = create_engine(f'sqlite:///{DB_PATH}')
    query = """
        SELECT Name, Sex, Age, BodyweightKg, TotalKg, Goodlift, Date
        FROM meets
        WHERE Equipment = 'Raw'
            AND Event = 'SBD'
            AND TotalKG IS NOT NULL
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

def compute_slope(values):
    # Fit a linear slope to a sequence of values, return 0 if fewer than 2 points
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    slope, _ = np.polyfit(x, values, 1)
    return slope

def engineer_features(df):
    rows = []

    for name, group in df.groupby('Name'):
        group = group.sort_values('Date').reset_index(drop=True)
        group = group.head(MAX_MEETS_PER_LIFTER)  

        if len(group) < 2:
            continue

        for i in range(1, len(group)):
            history = group.iloc[:i]
            target = group.iloc[i]

            totals = history['TotalKg'].values
            gls = history['Goodlift'].values
            bws = history['BodyweightKg'].values
            dates = history['Date']

            # Time features
            career_months = (
                dates.iloc[-1] - dates.iloc[0]
            ).days / 30.44 if len(dates) > 1 else 0

            gaps = [
                (dates.iloc[j+1] - dates.iloc[j]).days / 30.44
                for j in range(len(dates) - 1)
            ]

            avg_months_between = np.mean(gaps) if gaps else 0

            months_to_next = (
                target['Date'] - dates.iloc[-1]
            ).days / 30.44 

            # Skip bad time gaps
            if months_to_next <= 0 or months_to_next > 60:
                continue

            bw_change_last = (
                target['BodyweightKg'] - history['BodyweightKg'].iloc[-1]
            )
            if abs(bw_change_last) > 50:
                continue

            rows.append({
                # History aggregates
                'prev_total':         totals[-1],
                'best_total':         totals.max(),
                'mean_total':         totals.mean(),
                'total_slope':        compute_slope(totals),
                'prev_gl':            gls[-1],
                'best_gl':            gls.max(),
                'gl_slope':           compute_slope(gls),
                'num_meets':          len(history),
                'career_months':      career_months,
                'avg_months_between': avg_months_between,
                'bw_mean':            bws.mean(),
                'bw_std':             bws.std() if len(bws) > 1 else 0,
                'bw_change_last':     bw_change_last,
                # Target meet context
                'next_bodyweight':    target['BodyweightKg'],
                'months_to_next':     months_to_next,
                'age_at_next':        target['Age'],
                'Sex':                target['Sex'],
                # Target
                'next_total':         target['TotalKg'],
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

    le = LabelEncoder()
    transitions['Sex'] = le.fit_transform(transitions['Sex'])

    features = [
        'prev_total', 'best_total', 'mean_total', 'total_slope',
        'prev_gl', 'best_gl', 'gl_slope',
        'num_meets', 'career_months', 'avg_months_between',
        'bw_mean', 'bw_std', 'bw_change_last',
        'next_bodyweight', 'months_to_next',
        'age_at_next', 'Sex'
    ]

    X = transitions[features]
    y = transitions['next_total']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    model = XGBRegressor(
        n_estimators = 300,
        max_depth = 6,
        learning_rate = 0.05,
        random_state = 42,
        n_jobs = -1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"MAE: {mae:.2f} kg")
    print(f"R^2: {r2:.4f}")

    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print("Model saved")

    return model, le, X_test, y_test, preds, transitions

if __name__ == "__main__":
    train()