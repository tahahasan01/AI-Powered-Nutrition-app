import os
import re
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

DATA_PATH = "workoutdata_with_estimated_met.csv"
MODEL_PATH = "workout_model.pkl"

# Canonical feature names expected after normalization
CANONICAL_FEATURES = [
    "Primary_Muscle",
    "Equipment",
    "Difficulty",
    "Type",
    "Mechanics",
    "Level",
]
TARGET_COL = "MET"

# Map normalized header tokens -> canonical
ALIAS_MAP = {
    "title": "Exercise_Name",
    "exercisename": "Exercise_Name",
    "name": "Exercise_Name",

    "bodypart": "Primary_Muscle",
    "primarymuscle": "Primary_Muscle",
    "musclegroup": "Primary_Muscle",

    "equipment": "Equipment",

    "level": "Level",
    "difficulty": "Difficulty",
    "difficultylevel": "Difficulty",

    "desc": "Instructions",
    "description": "Instructions",
    "instructions": "Instructions",

    "type": "Type",
    "category": "Type",

    "mechanics": "Mechanics",
    "movementtype": "Mechanics",

    "caloriesburned": "MET",  # dataset uses this but values look like METs
    "burnedcalories": "MET",
    "burned_calories": "MET",
    "calories_burned": "MET",
    "met": "MET",
    "mets": "MET",
}


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    rename_map = {}
    for col in list(df.columns):
        key = norm(col)
        if key in ALIAS_MAP:
            canonical = ALIAS_MAP[key]
            if canonical not in df.columns:
                rename_map[col] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
    # If Difficulty is missing but Level exists, mirror it
    if "Difficulty" not in df.columns and "Level" in df.columns:
        df["Difficulty"] = df["Level"].astype(str)
    return df


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    df = normalize_headers(df)
    return df


def build_pipeline(categorical_features, numeric_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", random_state=42, max_iter=300, early_stopping=True)

    model = Pipeline(steps=[("preprocess", preprocessor), ("mlp", mlp)])
    return model


def main():
    df = load_dataset(DATA_PATH)

    # Determine available features among canonical ones
    available_features = [c for c in CANONICAL_FEATURES if c in df.columns]
    if not available_features:
        raise ValueError("No usable feature columns found after normalization.")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}' after normalization.")

    df = df.dropna(subset=available_features + [TARGET_COL]).copy()

    X = df[available_features]
    y = df[TARGET_COL].astype(float)

    categorical = available_features
    numeric = []

    model = build_pipeline(categorical_features=categorical, numeric_features=numeric)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Using features:", available_features)
    print("R2:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
