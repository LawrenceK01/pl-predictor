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
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_model.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.joblib")

def load_clean_data():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    query = """
        SELECT Sex, Age, BodyweightKg, TotalKg
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
    return pd.read_sql(query, engine)

def train():
    print("Loading clean data...")
    df = load_clean_data()

    # Encode Sex as numeric
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex']) # F=0, M=1

    # Add BodyweightKg^2 and Age^2 as features
    df['BodyweightKg^2'] = df['BodyweightKg'] ** 2
    df['Age^2'] = df['Age'] ** 2


    X = df[['Sex', 'Age', 'Age^2', 'BodyweightKg', 'BodyweightKg^2']]
    y = df['TotalKg']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"MAE: {mae:.2f} kg")
    print(f"R²: {r2:.4f}")

    # Save model and encoder
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print("Model and encoder saved.")

    return model, le, X_test, y_test, preds

if __name__ == "__main__":
    train()