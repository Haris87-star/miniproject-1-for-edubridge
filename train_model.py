"""
House Price Prediction - Model Training Script
EduBridge Mini Project 1

Trains and evaluates multiple regression models on housing data,
selects the best performing model, and serializes it to house_price_model.pkl.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years"]
MODEL_PATH = Path(__file__).resolve().parent / "house_price_model.pkl"


def generate_housing_data(n_samples: int = 500) -> pd.DataFrame:
    """Generate realistic synthetic housing dataset with reproducible random seed."""
    rng = np.random.default_rng(42)

    area_sqft = rng.integers(600, 4501, size=n_samples)
    bedrooms = rng.integers(1, 7, size=n_samples)
    bathrooms = rng.integers(1, 5, size=n_samples)
    age_years = rng.integers(0, 51, size=n_samples)

    # Realistic base price formula + random market variance
    base_price = 55_000
    price_per_sqft = 185
    bedroom_val = 14_000
    bathroom_val = 20_000
    depreciation_per_year = 1_200
    market_noise = rng.normal(0, 16_000, size=n_samples)

    price = (
        base_price
        + (area_sqft * price_per_sqft)
        + (bedrooms * bedroom_val)
        + (bathrooms * bathroom_val)
        - (age_years * depreciation_per_year)
        + market_noise
    )

    return pd.DataFrame(
        {
            "area_sqft": area_sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age_years,
            "price": np.round(price, 2),
        }
    )


def train_and_evaluate() -> None:
    print("=" * 60)
    print("House Price Prediction - Model Training (EduBridge Mini Project 1)")
    print("=" * 60)

    # 1. Load dataset
    df = generate_housing_data(500)
    print(f"[+] Generated dataset with {len(df)} samples and features: {FEATURES}")

    X = df[FEATURES]
    y = df["price"]

    # 2. Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[+] Split: {len(X_train)} training samples, {len(X_test)} testing samples")

    # 3. Candidate models
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
    }

    results = []

    print("\n--- Model Evaluation (Test Set) ---")
    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = root_mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        results.append(
            {
                "Model": name,
                "MAE ($)": round(mae, 2),
                "RMSE ($)": round(rmse, 2),
                "R2 Score": round(r2, 4),
            }
        )

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # 4. Select best model based on R2 score
    best_row = results_df.sort_values(by="R2 Score", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    print(f"\n[*] Best performing model: {best_model_name} (R2 = {best_row['R2 Score']})")

    # 5. Fit selected model on full dataset for maximum generalization in production
    final_model = candidate_models[best_model_name]
    final_model.fit(X, y)

    # 6. Save model bundle
    bundle = {
        "model": final_model,
        "features": FEATURES,
        "model_name": best_model_name,
        "metrics": {
            "r2_score": float(best_row["R2 Score"]),
            "mae": float(best_row["MAE ($)"]),
            "rmse": float(best_row["RMSE ($)"]),
        },
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f"[+] Successfully serialized model to: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train_and_evaluate()
