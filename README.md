# Forecasting Real-Time Energy Demand
### Data Science Final Project — Ramson Munoz & Valentina Kloster

---

## Project Summary

This project analyses hourly electricity market data from **ISO New England (2018–2025)**
and builds a **Random Forest model** to forecast Real-Time electricity demand (RT_Demand)
one day ahead. The analysis covers the full data science pipeline:
Exploratory Data Analysis → Hypothesis Testing → Machine Learning → Interactive Dashboard.

**Key result:** A single 24-hour lag feature achieves a MAPE of ~3–5% on the held-out
2024–2025 test set — competitive with operational short-term forecasting benchmarks.

---

## Repository Structure

```
project/
├── data/
│   └── combined_data_hourly.csv   ← main dataset (2016–2025 hourly)
├── dashboard.py                   ← Streamlit application (run this)
├── report.pdf                     ← written report
└── README.md                      ← this file
```

---

## Requirements

Python 3.9+ is recommended. Install all dependencies with:

```bash
pip install streamlit pandas plotly scikit-learn statsmodels scipy matplotlib
```

---

## How to Run the Streamlit App

1. Make sure `combined_data_hourly.csv` is inside a `data/` folder in the same
   directory as `dashboard.py`.

2. From your terminal, navigate to the project folder:
   ```bash
   cd path/to/project
   ```

3. Launch the app:
   ```bash
   streamlit run dashboard.py
   ```

4. Your browser will open automatically at `http://localhost:8501`.

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
- **Rows (filtered):** ~61,368 (2018-01-01 to 2025-12-31)
- **Columns:** 14 (Date, Hr_End, DA/RT Demand, DA/RT LMP components, Dry_Bulb, Dew_Point)
- **Missing values:** None

---

## Notes

- The ML tab runs cross-validation (3–8 folds, adjustable via slider) across 12 parameter
  combinations. On a standard laptop this takes **1–3 minutes** the first time. Results are
  cached by Streamlit after the first run within the same session.
- The train/test split is strictly chronological: training on 2018–2023, testing on 2024–2025.
- The 24-hour lag shifts RT_Demand back by 24 rows; the first 24 rows are dropped to avoid NaN.
