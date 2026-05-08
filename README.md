# NYC Sanitation Burden Explorer

Combined final project for **CUSP Spring 2026 Civic Analytics** and **Applied Data Science**.

## Team
- Akbar Abilov
- Ayla Fish
- U.P. Nguyen
- Arya Roi

## Project overview
This project examines sanitation-related 311 complaints in New York City to identify **persistent complaint burden**, place that burden in **demographic and socioeconomic context**, and build an **exploratory predictive layer** using Bayesian modeling. The repository combines a reproducible data pipeline, an interactive Streamlit dashboard, and an optimization component.

## Research question
**Which NYC neighborhoods have persistent sanitation-related complaint burden, and how does that compare with neighborhood context?**

## Project components

### Descriptive / dashboard layer
- Cleaned 311 sanitation complaints dataset
- Monthly neighborhood-level aggregation by geography
- Merged demographic context
- Interactive dashboard with:
  - burden maps
  - monthly trends
  - demographic comparisons
  - overlap flags

### Bayesian modeling layer
- Bayesian hierarchical negative binomial model
- Monthly complaint burden estimated at the **community district x month** level
- Posterior prediction intervals
- Observed-vs-expected mismatch / anomaly framing
- Districts with highest predicted burden and highest uncertainty

### Optimization layer
- K-means clustering and optimization logic for thinking about where sanitation trucks could be allocated based on burden
- Optimization code is stored in `app.py`

---

## Repository structure

```text
solid-waste-civic-tech/
├── app/                  # Streamlit dashboard
├── data/
│   ├── raw/              # Raw downloaded files
│   └── processed/        # Cleaned / modeled outputs
├── src/                  # Python data pipeline and modeling scripts
├── requirements.txt
└── README.md
```

Our process:
- raw downloaded data is written to `data/raw/`
- processed aggregated files go to `data/processed/`
- pipeline scripts live in `src/`
- the dashboard lives in `app/`

---

## Data source
The project uses the **NYC Open Data 311 endpoint (2020–present)**:

```python
BASE_311_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
```

This project specifically focuses on sanitation-related complaints, including illegal dumping and missed collection.

---

## Setup

From the project root:

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Optional sanity check:

```bash
python -c "import numpy, pandas, pyarrow, matplotlib, arviz, pymc, bambi, streamlit, plotly, folium; print('ok')"
```

---

## How to run the project


## Demographic data generation

To generate the demographics CSV from Census / ACS data:

```bash
python src/generate_demographics_from_acs.py --out data/processed/demographics_placeholder.csv
```

### Step 1: Extract and process 311 data
This creates the demographic CSV used in the merge step.

```bash
python src/extract_311.py
python src/build_features.py
python src/merge_demo.py
python src/build_dashboard_data.py
```

### Step 2: Build the Bayesian modeling dataset

```bash
python src/build_bayesian_dataset.py
```

### Step 3: Fit the Bayesian model

```bash
python src/fit_bayesian_model.py
```

### Step 4: Generate Bayesian predictions

```bash
python src/predict_bayesian_model.py
```

### Step 5: Launch the Streamlit dashboard

```bash
python -m streamlit run app/explorer.py
```

---

## Bayesian modeling notes

Our Bayesian model is best interpreted as an **exploratory predictive model of expected monthly 311 complaint burden**, not as a direct measure of on-the-ground trash conditions. Based on project feedback, we frame the output as an **observed-vs-expected mismatch** or **anomaly screen**, rather than as proof of underreporting.

### Model summary
We implemented a **Bayesian hierarchical negative binomial model** to estimate monthly sanitation complaint burden by community district, accounting for:
- population exposure
- temporal persistence
- seasonality
- district-level heterogeneity

### Why Bayesian?
Two main reasons:
1. **Small or noisy samples**  
   Bayesian estimation is less likely to overreact to one unusual month in sparse complaint categories.
2. **Partial pooling across districts**  
   Community districts can borrow strength from broader borough and city-level patterns, improving stability where data are limited.

### Interpretation guidance
- The model predicts **expected monthly 311 complaint burden**
- Observed complaints are compared against the model's expected range
- Districts outside the expected interval are treated as **mismatches for follow-up**
- These mismatches should be interpreted as **screening signals**, not definitive evidence of underreporting

### Current modeling status
The model fit improved substantially after tuning, and that prior complaint burden and seasonality emerged as strong predictors. They also note that a small number of divergences remained, so the model should be treated as an **exploratory predictive layer rather than a definitive inferential model**.

---

## Current dashboard features

### Descriptive layer
- Burden map
- Trends
- Demographic comparisons
- Overlap flags

### Bayesian layer
- Predicted complaints
- Posterior interval
- Districts with highest predicted burden
- Districts with highest uncertainty

---

## Deliverables / outputs
Version 1 project outputs:
- a cleaned 311 sanitation complaints dataset
- a neighborhood-level monthly aggregation
- a merged dataset with demographic context
- early visualizations such as complaint hotspots, complaints per capita, and repeat complaints over time
- a simple dashboard in Streamlit or Plotly Dash

---

## Notes and limitations
- 311 complaints reflect **reported complaint activity**, not direct observation of sanitation conditions
- The Bayesian model estimates **expected complaint burden**, not true trash conditions
- Any observed-vs-expected mismatch should be treated as a prompt for further investigation.
- Proving "under-reporting" is outside the scope of this project, given the time and constraints we have. 
- Demographic/contextual data are merged in to support interpretation, not to imply simple causal conclusions

---

## Submission note
This repository is intended as the combined final project for the Spring 2026 Civic Analytics and Applied Data Science courses.
