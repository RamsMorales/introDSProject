import pandas as pd  # Pandas for data manipulation
import streamlit as st  # Streamlit library for web apps (dashboards for our data)
import plotly.express as px  # Plotly Express for creating charts
from math import sqrt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

import matplotlib.pyplot as plt

st.title("Final Project")  # Title of the dashboard
st.header("Ramson Munoz & Valentina Kloster")  # Subtitle of the dashboard
st.subheader("Forcasting real time energy demand")  # Header of the dashboard

# To run: streamlit run dashboard.py

st.divider()


df = pd.read_csv("data/combined_data_hourly.csv")
# we just want to include the records from 2018 to 2025
MIN_DATE = "2018-01-01"
MAX_DATE= "2025-12-31"
df = df[df["Date"].between(MIN_DATE, MAX_DATE)]
## Dropping duplicates
if "02X" in df["Hr_End"].values.astype(str):
    #st.write("True")
    df.drop(df.loc[df["Hr_End"].astype(str)=="02X"].index,inplace=True)

## Converting HR_End variable to numeric to make Date_Time feature for time series
df["Hr_End"] = df["Hr_End"].astype(int)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Preliminary Data Analysis",
        "EDA",
        "Key findings",
        "Hypothesis Testing",
        "ML Forecast",
    ]
)

with tab1:  # Preliminary Data Analysis
    st.subheader("Preliminary Data Analysis")
    st.write("First let's take a look at the data")
    st.write(df.head())

    st.write("Number of rows and columns:")
    st.write(df.shape)

    st.write("Data types:")
    st.write(df.dtypes)

    st.write("Description of each variable:")
    st.write(df.describe())

with tab2: # EDA
        st.subheader("EDA")
        with st.expander("Box Plot"):
            col1, col2 = st.columns([1,4]) # two columns, first one is 1/5 of the width, second one is 4/5 of the width
            with col1: 
                numeric_columns = df.select_dtypes(include="number").columns
                feature_selector = st.selectbox("Select a parameter to visualize", numeric_columns, key="feature_selector") # user selection
            with col2:
                fig1 = px.box(df, x=f"{feature_selector}", title=f"Box Plot of {feature_selector}")
                st.plotly_chart(fig1)

        with st.expander("Histogram"):
            col1, col2 = st.columns([1,4]) # two columns, first one is 1/5 of the width, second one is 4/5 of the width
            with col1:
                # remove data and Hr_End from the parameter options.
                filtered_columns = df.columns.difference(["Date", "Hr_End"])
                parameter_selected2 = st.selectbox("Select a parameter to visualize", filtered_columns, key="parameter_selected2") # user selection
            with col2:
                fig2 = px.histogram(df,
                                    x=parameter_selected2,
                                    nbins=3,
                                    title=f"Histogram of {parameter_selected2}")
                st.plotly_chart(fig2)

        with st.expander("Heatmap"):
            df_numeric = df.select_dtypes(include=['number']) # select only numeric columns
            fig4 = px.imshow(df_numeric.corr(),
                            text_auto=True,
                            title="Heatmap of Correlation Matrix")
            st.plotly_chart(fig4)        

with tab3:  # key findings
    ## Plot of daily resolution
    # TODO: fix vanishing plot for comparison
    st.subheader("RT_Demand Shows seasonal behavior on Daily  and Yearly timescale")

    with st.expander("Daily Resolution"):
        date_selected = st.date_input("Choose a date to visualize",
                                                                value="2018-01-01",
                                                                min_value=MIN_DATE,
                                                                max_value=MAX_DATE,
                                                                key="Daily_Res") 
        start = st.button("Start", key="daily")
        if start:
            dates = df[df["Date"] == date_selected.strftime('%Y-%m-%d')]
            # NOTE: shifted end of hour to start of hour to ensure 24 hours remained in the same date 
            dates['DateTime'] = pd.to_datetime(dates['Date']) + pd.to_timedelta(dates['Hr_End']-1, unit='h') 
            daily_ts_fig = px.line(dates,
                                                x="DateTime",
                                                y="RT_Demand",
                                                title=f"Hourly RT_Demand for {date_selected.strftime('%m-%d-%Y')}"
                                                )
            st.plotly_chart(daily_ts_fig)
    with st.expander("Weekly Resolution"):
        left_date, right_date = st.date_input("Choose a date range to visualize",
                                                                value=("2018-01-01","2018-01-08"),
                                                                min_value=MIN_DATE,
                                                                max_value=MAX_DATE,
                                                                key="Weekly_Res") 
        start2  = st.button("Start", key= "weekly")
        if start2:
            dates2 = df[df["Date"].between(left=left_date.strftime('%Y-%m-%d'),right=right_date.strftime('%Y-%m-%d'),inclusive="both")]
            dates2['DateTime'] = pd.to_datetime(dates2['Date']) + pd.to_timedelta(dates2['Hr_End']-1, unit='h') 
            
            weekly_ts_fig = px.line(dates2,
                                                x="DateTime",
                                                y="RT_Demand",
                                                title=f"RT_Demand for {left_date.strftime('%m-%d-%Y')} to {right_date.strftime('%m-%d-%Y')}"
                                                )
            st.plotly_chart(weekly_ts_fig)
            st.caption("Date ranges on shorter time scales reveal weekly seasonality in addtion to the daily seasonality evident in the previous figures.")
    with st.expander("Yearly Resolution"):
        selected_year = st.selectbox("Select a year to visualize",pd.to_datetime(df["Date"]).dt.year.unique())
        dates3 = df[df["Date"].str.contains(str(selected_year))] 

        yearly_RT_Demand_Smoothed = dates3.groupby("Date")["RT_Demand"].mean().reset_index()

        yearly_ts_fig = px.line(yearly_RT_Demand_Smoothed,
                                            x="Date",
                                            y="RT_Demand",
                                            title=f"Average daily RT_Demand for {selected_year}"
                                            )
        st.plotly_chart(yearly_ts_fig)
        st.caption("Yearly plot reveals both monthly and yearly seasonality in RT_Demand")

with tab4: # hypothesis testing
    st.subheader("Is there Stationarity for hourly RT_Demand for the years 2018 through 2025?")
    st.divider()
    st.write("Our goal is to predict next day RT_Demand by hour using a Random Forrest Time series" \
    " approach. In the Key Findings tab, we saw evidence of periodic behavior for RT demand" \
    " on the hourly and yearly time scale, with weak evidence for periodic behavior for the weekly" \
    " time scale. To examine the design of the feature lag variables, we will use the augmented Dickey" \
    "-Fuller and the Kwiatkowski–Phillips–Schmidt–Shin test to check for trend stationarity. Although," \
    " these test are not strictly necessary for random forrest, they do help us decide the lag differencing" \
    " in the model.")
    with st.container(border=True):
        st.subheader("Dickey-Fuller test")
        st.latex(r"\mathbf{H_0} :  \mathrm{The\ time\ series\ is\ non-stationarity\ i.e\ }  \exists \mathrm{\ a\ unit\ root} ")
        st.latex(r"\mathbf{H_a} :  \mathrm{The\ time\ series\ is\ stationarity\ i.e\ }  \nexists \mathrm{\ a\ unit\ root} ")
        adf_result = adfuller(df["RT_Demand"], autolag="AIC")
        st.write(f"ADF Statistic: {adf_result[0]}")
        st.write(f"p-value: {adf_result[1]}")
        st.write("Assuming significance of 0.05, the p-value indicates that the data is stationary." \
        " Meaning that there are no trends in the data over the yearly time span.")
    
    with st.container(border=True):
        st.subheader("KPSS test")
        st.latex(r"\mathbf{H_0} :  \mathrm{The\ time\ series\ is\ stationarity\ i.e\ }  \nexists \mathrm{\ a\ unit\ root} ")
        st.latex(r"\mathbf{H_a} :  \mathrm{The\ time\ series\ is\ non-stationarity\ i.e\ }  \exists \mathrm{\ a\ unit\ root} ")
        kpss_result = kpss(df["RT_Demand"], regression="c",nlags="auto")
        st.write(f"KPSS Statistic: {kpss_result[0]}")
        st.write(f"p-value: {kpss_result[1]}")
        st.write("Assuming significance of 0.05, the p-value indicates that the data is stationary. "\
        " Meaning that there are no trends in the data over the yearly time span.")
    
    st.divider()
    st.write("The results indicate that trends are not present in the data, but our plots indicate a " \
    "seasonal pattern which may be important for determining our lags. To address this, we will " \
    "decompose the data to examine the patterns.")
    decomposition = seasonal_decompose(df["RT_Demand"],model="additive",period=24)
    st.pyplot(decomposition.plot())

with tab5:  # ML forecast
    st.subheader("Random Forest Forecast")
    st.write(
        "We create a 24-hour lag feature, use time-series cross validation on the training set, "
        "and choose the parameter combination with the lowest average MAE."
    )
    lag =24
    df_lagged = df.copy()
    # Build an hourly timestamp so the rows can be ordered correctly before creating lags.
    # This is important because shift(24) means "24 rows earlier", so the dataframe must
    # be in true time order for the lag to represent the same hour on the previous day.
    df_lagged["DateTime"] = pd.to_datetime(df_lagged["Date"]) + pd.to_timedelta(
        df_lagged["Hr_End"] - 1, unit="h"
    )
    df_lagged = df_lagged.sort_values("DateTime")

    # Use demand from the same hour one day earlier as the predictor.
    df_lagged["RT_Demand-24"] = df_lagged["RT_Demand"].shift(lag)
    # The first 24 rows have no previous-day value, so we remove them.
    df_lagged = df_lagged.dropna()

    # Train on data before 2024 and test on 2024 onward.
    # This keeps the split chronological, which is required for time-series modeling.
    training_data = df_lagged[df_lagged["Date"] < "2024-01-01"].copy()
    test_data = df_lagged[df_lagged["Date"] >= "2024-01-01"].copy()

    # For now the model uses only one feature: yesterday's demand at the same hour.
    X_train_full = training_data[["RT_Demand-24"]]
    y_train_full = training_data["RT_Demand"]
    X_test = test_data[["RT_Demand-24"]]
    y_test = test_data["RT_Demand"]

    st.markdown("**Training/Test Split**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Training rows", f"{len(training_data):,}")
    col2.metric("Test rows", f"{len(test_data):,}")
    col3.metric("Lag used", f"{lag} hours")

    # Parameter grid to try during cross validation.
    # You can expand these lists later if you want a wider search.
    n_splits = st.slider("Number of CV folds", min_value=3, max_value=8, value=5, step=1)
    n_estimators_list = [100, 200, 300]
    max_depth_list = [5, 10, 15, None]

    # TimeSeriesSplit preserves time order:
    # earlier data is used for training, later data for validation.
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = []
    best_mae = float("inf")
    best_params = {}

    # Try every parameter combination and compute validation MAE on each fold.
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            fold_maes = []
            fold_rmses = []
            fold_mapes = []

            for fold_number, (train_index, val_index) in enumerate(tscv.split(X_train_full), start=1):
                X_train = X_train_full.iloc[train_index]
                X_val = X_train_full.iloc[val_index]
                y_train = y_train_full.iloc[train_index]
                y_val = y_train_full.iloc[val_index]

                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                fold_mae = mean_absolute_error(y_val, y_pred)
                fold_rmse = sqrt(mean_squared_error(y_val, y_pred))
                fold_mape = mean_absolute_percentage_error(y_val, y_pred) * 100
                fold_maes.append(fold_mae)
                fold_rmses.append(fold_rmse)
                fold_mapes.append(fold_mape)

                # Save each fold result so it can be shown later in Streamlit.
                cv_results.append(
                    {
                        "n_estimators": n_estimators,
                        "max_depth": "None" if max_depth is None else max_depth,
                        "fold": fold_number,
                        "mae": fold_mae,
                        "rmse": fold_rmse,
                        "mape": fold_mape,
                    }
                )

            # Average the fold errors for this parameter combination.
            avg_mae = sum(fold_maes) / len(fold_maes)
            if avg_mae < best_mae:
                best_mae = avg_mae
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}

    cv_results_df = pd.DataFrame(cv_results)
    # Summarize one row per parameter combination for the table and chart.
    summary_df = (
        cv_results_df.groupby(["n_estimators", "max_depth"], as_index=False)[["mae", "rmse", "mape"]]
        .mean()
        .rename(columns={"mae": "avg_mae", "rmse": "avg_rmse", "mape": "avg_mape"})
        .sort_values("avg_mae")
    )
    best_summary_row = summary_df.iloc[0]

    st.markdown("**Cross Validation Results**")
    st.dataframe(summary_df, use_container_width=True)

    cv_chart = px.bar(
        summary_df,
        x="n_estimators", # n_estimators means number of trees in the random forest, which is a common parameter to tune for this model.
        y="avg_mae", 
        color="max_depth",
        barmode="group",
        title="Average MAE by Parameter Combination",
        labels={"avg_mae": "Average MAE", "n_estimators": "Number of Trees", "max_depth": "Max Depth"},
    )
    st.plotly_chart(cv_chart, use_container_width=True)

    st.markdown("**Best Parameters**")
    st.write(
        {
            "n_estimators": best_params["n_estimators"],
            "max_depth": "None" if best_params["max_depth"] is None else best_params["max_depth"],
            "cv_mae": round(best_summary_row["avg_mae"], 2),
            "cv_rmse": round(best_summary_row["avg_rmse"], 2),
            "cv_mape_percent": round(best_summary_row["avg_mape"], 2),
        }
    )

    st.markdown("**Validation Performance (Average Across Folds)**")
    val_col1, val_col2, val_col3 = st.columns(3)
    val_col1.metric("Validation MAE", f"{best_summary_row['avg_mae']:,.2f}")
    val_col2.metric("Validation RMSE", f"{best_summary_row['avg_rmse']:,.2f}")
    val_col3.metric("Validation MAPE", f"{best_summary_row['avg_mape']:,.2f}%")

    # Refit the model on the full training set using the winning parameters.
    final_model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_train_full, y_train_full)
    test_predictions = final_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_rmse = sqrt(mean_squared_error(y_test, test_predictions))
    test_mape = mean_absolute_percentage_error(y_test, test_predictions) * 100

    # Store actual and predicted values together for plotting.
    predictions_df = test_data[["DateTime", "RT_Demand"]].copy()
    predictions_df["Predicted_RT_Demand"] = test_predictions

    st.markdown("**Test Set Performance**")
    test_col1, test_col2, test_col3 = st.columns(3)
    test_col1.metric("Test MAE", f"{test_mae:,.2f}")
    test_col2.metric("Test RMSE", f"{test_rmse:,.2f}")
    test_col3.metric("Test MAPE", f"{test_mape:,.2f}%")

    # Show only the first 14 test days so the chart stays readable.
    forecast_chart = px.line(
        predictions_df.head(24 * 14),
        x="DateTime",
        y=["RT_Demand", "Predicted_RT_Demand"],
        title="Actual vs Predicted RT_Demand (First 14 Days of Test Set)",
        labels={"value": "RT_Demand", "variable": "Series"},
    )
    st.plotly_chart(forecast_chart, use_container_width=True)


### TODO Add an Interpretation of the Results section

### TODO Add an Improvements and Next Steps section: here we can say that to improve the model, we would add more features such as day of week, month, and additional lags. We could also try other models such as XGBoost or LSTM neural networks. Additionally, we could expand the hyperparameter search to include more values and parameters.

### TODO Create the report and a README file to explain how to run the code and summarize the results.
