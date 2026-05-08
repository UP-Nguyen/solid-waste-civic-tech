# solid-waste-civic-tech
Combined project for CUSP Spring 2026 Civic Analytics and Applied Data Science


# NYC Open Data 311 endpoint (2020-present)
BASE_311_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

# This is our README for our combined ADS and Civic Analytics Final Project!

Version 1:
Question:
Which NYC neighborhoods have persistent sanitation-related complaint burden, and how does that compare with neighborhood context?

v1 outputs:
A cleaned 311 sanitation complaints dataset
A neighborhood-level aggregation by month + geography
A merged dataset with demographic context
2–3 first visualizations:
complaint hotspots
complaints per capita
repeat complaints over time
A simple dashboard in Streamlit or Plotly Dash
0

What goes where:

Raw downloaded data gets written to data/raw/
Processed aggregated files go to data/processed/
Python pipeline scripts live in src/
The dashboard lives in app/


# How to run this project

# From root folder 
python -m venv .venv
source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
#optional for quick sanity check and make sure requirements have been installed properly
python -c "import numpy, pandas, pyarrow, matplotlib, arviz, pymc, bambi, streamlit, plotly, folium; print('ok')"

python src/extract_311.py
python src/build_features.py
python src/merge_demo.py          
python src/build_dashboard_data.py


--create a modeling dataset
python src/build_bayesian_dataset.py

--fit the Bayesian model with Bambi
python src/fit_bayesian_model.py

--generate predictions for the latest month + next month
python src/predict_bayesian_model.py

python -m streamlit run app/explorer.py

# To get demographics csv from census
python src/generate_demographics_from_acs.py --out data/processed/demographics_placeholder.csv

# Bayes Modeling

Based on feedback from our Applied Data Presentation, we wanted to note that our Bayes Model does not predict expected burden. Instead, it estimates expected monthly complaint burden. Phrased differently, it predicts "what might be" 311 complaints. We then compare observed complaints to that expected range. Districts outside the model’s expected interval are treated as observed-vs-expected mismatches or anomalies for follow-up, not as proof of underreporting.

Because sanitation complaint data are noisy, uneven across districts, and subject to overdispersion, a Bayesian hierarchical count model may be more appropriate than a simple linear model. This approach allows estimates for individual districts to be partially pooled toward broader borough- and city-level patterns, reducing instability in low-data areas while still preserving local variation. It also provides credible intervals that reflect uncertainty in predicted complaint burden.

Why Bayesian modeling?
1. Small or noisy samples
If some complaint categories are sparse, a Bayesian model can avoid overreacting to one weird month.

2. Partial pooling across districts
This is a big one. Instead of treating every community district as fully independent, we can let districts “borrow strength” from the overall city pattern and from their borough. (some districts have a lot of data, and some have less)

Install the Bayesian packages (see requirements)


# To run the Bayes Model
create a modeling dataset
python src/build_bayesian_dataset.py

fit the Bayesian model with Bambi
python src/fit_bayesian_model.py

generate predictions for the latest month + next month
python src/predict_bayesian_model.py

We implemented a Bayesian hierarchical negative binomial model to estimate monthly sanitation complaint burden by community district, accounting for population exposure, temporal persistence, seasonality, and district-level heterogeneity. The model fit improved substantially after tuning, and preliminary results suggest prior complaint burden and seasonality are strong predictors.

The Bayesian model converged reasonably well after tuning, though a small number of divergences remained. Results are therefore used as an exploratory predictive layer rather than a definitive inferential model.

Currently:
current dashboard as the descriptive layer:
burden map
trends
demographic comparisons
overlap flags

Then the Bayesian prediction layer:

predicted complaints next month
posterior interval
districts with highest predicted burden
districts with highest uncertainty

# We also built an optimization model with k means clustering that can optimize where garbage trucks could be allocated, based on burden. That code can be found in  

solid-waste-civic-tech/app.py