# Forecasting Real-Time Energy Demand
### Data Science Final Project — Ramson Munoz & Valentina Kloster

---

## Project Summary

This project analyses hourly electricity market data from **ISO New England (2018–2025)**
and builds a **Random Forest model** to forecast Real-Time electricity demand (RT_Demand)
one day ahead. The analysis covers the full data science pipeline:
Exploratory Data Analysis → Hypothesis Testing → Machine Learning → Interactive Dashboard.

**Key result:** A model using 24h and 168h lag features plus a Month variable achieves a
MAPE of ~3–5% on the held-out 2024–2025 test set, competitive with operational short-term
forecasting benchmarks.

---

## Repository Structure

```
INTRODSP.../
├── data/
│   ├── 2018_smd_hourly.xlsx         ← raw ISO-NE data per year
│   ├── 2019_smd_hourly.xlsx
│   ├── 2020_smd_hourly.xlsx
│   ├── 2021_smd_hourly.xlsx
│   ├── 2022_smd_hourly.xlsx
│   ├── 2023_smd_hourly.xlsx
│   ├── 2024_smd_hourly.xlsx
│   ├── 2025_smd_hourly.xlsx
│   └── combined_data_hourly.csv     ← combined dataset used by the app
├── venvds/                          ← virtual environment (do not commit)
├── .gitignore
├── dashboard.py                     ← Streamlit application (run this)
├── README.md                        ← this file
└── requirements.txt                 ← all Python dependencies
```

---

## Requirements

Python 3.9+ is recommended. Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## How to Run the Streamlit App

1. Make sure you are in the project root folder (where `dashboard.py` lives):
   ```bash
   cd path/to/INTRODSP
   ```

2. (Optional but recommended) Activate the virtual environment:
   ```bash
   # Mac / Linux
   source venvds/bin/activate

   # Windows
   venvds\Scripts\activate
   ```

3. Install dependencies if you haven't already:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the app:
   ```bash
   streamlit run dashboard.py
   ```

5. Your browser will open automatically at `http://localhost:8501`.

---

## Dashboard Tabs

| Tab | Contents |
|-----|----------|
| **Preliminary Data Analysis** | Shape, dtypes, missing values, descriptive statistics |
| **EDA** | Interactive box plot, histogram (50 bins), correlation heatmap |
| **Key Findings** | RT_Demand time series at daily, weekly, and yearly resolution |
| **Hypothesis Testing** | ADF test, KPSS test, seasonal decomposition, Summer vs. Winter t-test |
| **ML Forecast** | Random Forest with time-series CV, best params, test performance, Actual vs Predicted chart, interpretation, next steps |

---

## Dataset

- **Source:** [ISO New England SMD Hourly](https://www.iso-ne.com)
- **Raw files:** one Excel file per year (2018–2025), stored in `data/`
- **Combined file:** `combined_data_hourly.csv` merges all years into a single CSV
- **Rows (filtered):** ~61,368 (2018-01-01 to 2025-12-31)
- **Columns:** 14 (Date, Hr_End, DA/RT Demand, DA/RT LMP components, Dry_Bulb, Dew_Point)
- **Missing values:** None

---

## Notes

- The ML tab runs cross-validation (3–8 folds, adjustable via slider) across 12 parameter
  combinations. On a standard laptop this takes **1–3 minutes** the first time. Results are
  cached by Streamlit after the first run within the same session.
- The train/test split is strictly chronological: training on 2018–2023, testing on 2024–2025.
- The model uses three features: a 24-hour lag, a 168-hour lag (same hour one week prior),
  and a Month variable. The first 168 rows are dropped since they have no valid lag values.
- The `venvds/` folder should not be committed to version control — it is listed in `.gitignore`.