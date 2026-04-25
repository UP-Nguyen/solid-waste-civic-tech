# solid-waste-civic-tech
Combined project for CUSP Spring 2026 Civic Analytics and Applied Data Science

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


How to run:

# From root folder 
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt


python src/extract_311.py
python src/build_features.py
streamlit run app/streamlit_app.py


# NOTES
map? Tableau? 
Track basic performance metrics
Animated map/scrollable that shows

1 map that identifies still visual
Another map that shows how persistent/systemic patterns

#4/13/26
1. Animated choropleth of complaints per 1,000 residents 
2. Animated scatterplot of complaints per capita vs median income
- This supports the equity framing without making causal claims. 
3. Animated ranking chart of top community districts by repeat complaint burden 
 -This helps define “persistent inefficiency” more concretely.

from main folder to run
python src/extract_311.py
python src/build_features.py
python src/merge_demo.py
streamlit run app/animated_visuals.py

community district code
total population
median household income

# 4/25/26
pip install -r requirements.txt
python src/extract_311.py
python src/build_features.py
python src/merge_demo.py          # optional but recommended
python src/build_dashboard_data.py
streamlit run app/explorer.py