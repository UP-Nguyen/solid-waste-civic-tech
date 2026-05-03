from __future__ import annotations

from pathlib import Path
#import pickle

import arviz as az
import bambi as bmb
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "processed" / "bayes_model_input.parquet"
MODEL_DIR = BASE_DIR / "data" / "processed" / "bayes_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_parquet(INPUT_FILE).copy()

    # Treat seasonality as categorical
    df["month_num"] = df["month_num"].astype("category")
    df["borough"] = df["borough"].astype("category")
    df["community_board"] = df["community_board"].astype("category")

    # Hierarchical negative binomial with random intercepts and offset
    formula = (
        "complaints ~ median_income_k + poverty_rate + lag_complaints_log1p "
        "+ month_num + (1|borough) + (1|community_board)"
    )

    formula = (
        "complaints ~ median_income_k + poverty_rate + lag_complaints_log1p "
        "+ month_num + (1|borough) + (1|community_board) + offset(log_population)"
    )

    model = bmb.Model(
        formula,
        df,
        family="negativebinomial",
    )

    idata = model.fit(
        draws=1000,
        tune=1500,
        chains=4,
        cores=4,
        target_accept=0.99,
        init="adapt_diag",
        random_seed=42,
        nuts={"max_treedepth": 18},
    )

    # Save inference data
    az.to_netcdf(idata, MODEL_DIR / "bayes_model_idata.nc")

    # Save a summary table
    summary = az.summary(idata)
    summary.to_csv(MODEL_DIR / "bayes_model_summary.csv")

     # Save model object for later predictions
    #with open(MODEL_DIR / "bambi_model.pkl", "wb") as f:
    #    pickle.dump(model, f) 

    print("Saved model outputs to:", MODEL_DIR)
    print(summary.head(20))


if __name__ == "__main__":
    main()