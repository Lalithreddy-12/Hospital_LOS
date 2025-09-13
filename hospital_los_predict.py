#!/usr/bin/env python3
"""
Hospital LOS Prediction + XAI
Usage:
    python hospital_los_predict.py --data data/hospital_los.csv --target LOS --output out/
"""
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def parse_args():
    p = argparse.ArgumentParser(description="Hospital LOS prediction + XAI")
    p.add_argument("--data", required=True, help="Path to CSV dataset")
    p.add_argument("--target", default="LOS", help="Target column name (numeric LOS)")
    p.add_argument("--output", default="out", help="Output folder to save models/plots")
    p.add_argument("--test-size", default=0.2, type=float, help="Test split fraction")
    p.add_argument("--random-state", default=42, type=int)
    return p.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.data)
    y = df[args.target]
    X = df.drop(columns=[args.target])

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    pipeline = Pipeline(steps=[('preproc', preprocessor),
                               ('model', RandomForestRegressor(n_estimators=100, random_state=args.random_state))])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds))
    }

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "evaluation.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    joblib.dump(pipeline, os.path.join(args.output, "model.joblib"))
    print("Training complete. Metrics saved. Model saved.")

if __name__ == "__main__":
    main()
