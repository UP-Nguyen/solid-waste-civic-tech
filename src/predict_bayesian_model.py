from __future__ import annotations

from pathlib import Path

import arviz as az
import bambi as bmb
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "processed" / "bayes_model_input.parquet"
MODEL_DIR = BASE_DIR / "data" / "processed" / "bayes_model"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "bayes_predictions.parquet"


def main() -> None:
    df = pd.read_parquet(INPUT_FILE).copy()
    df["month_dt"] = pd.to_datetime(df["month"], format="%Y-%m")
    df["month_num"] = df["month_num"].astype("category")
    df["borough"] = df["borough"].astype("category")
    df["community_board"] = df["community_board"].astype("category")

    # Must exactly match fit_bayesian_model.py
    formula = (
        "complaints ~ median_income_k + poverty_rate + lag_complaints_log1p "
        "+ month_num + (1|borough) + (1|community_board) + offset(log_population)"
    )

    model = bmb.Model(
        formula,
        df,
        family="negativebinomial",
    )

    idata = az.from_netcdf(MODEL_DIR / "bayes_model_idata.nc")

    pred_idata = model.predict(
        idata=idata,
        data=df,
        kind="response_params",
        include_group_specific=False,
        inplace=False,
    )

    if "mu" not in pred_idata.posterior:
        raise ValueError("Expected posterior['mu'] was not found.")

    yhat = pred_idata.posterior["mu"]

    pred_median = yhat.quantile(0.50, dim=("chain", "draw")).values
    pred_low = yhat.quantile(0.10, dim=("chain", "draw")).values
    pred_high = yhat.quantile(0.90, dim=("chain", "draw")).values

    out = df.copy()
    out["predicted_complaints"] = pred_median
    out["predicted_complaints_p10"] = pred_low
    out["predicted_complaints_p90"] = pred_high
    out["prediction_interval_width"] = out["predicted_complaints_p90"] - out["predicted_complaints_p10"]
    out["prediction_error"] = out["complaints"] - out["predicted_complaints"]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_FILE, index=False)

    print(f"Saved predictions to {OUTPUT_FILE}")
    print(
        out[
            [
                "month",
                "borough",
                "community_board",
                "complaints",
                "predicted_complaints",
                "predicted_complaints_p10",
                "predicted_complaints_p90",
                "prediction_interval_width",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()