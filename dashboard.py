import pandas as pd  # Pandas for data manipulation
import streamlit as st  # Streamlit library for web apps (dashboards for our data)
import plotly.express as px  # Plotly Express for creating charts
from math import sqrt
from statsmodels.tsa.stattools import adfuller 
from pmdarima.arima.utils import nsdiffs
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV 
from sklearn.feature_selection import SelectFromModel
from scipy import stats  # For t-test (hypothesis testing)

import matplotlib.pyplot as plt

st.title("Final Project")  # Title of the dashboard
st.header("Ramson Munoz & Valentina Kloster")  # Subtitle of the dashboard
st.subheader("Forcasting real time energy demand")  # Header of the dashboard

# To run: streamlit run dashboard.py

st.divider()


MIN_DATE = "2018-01-01"
MAX_DATE= "2025-12-31"

@st.cache_data  # Cache so the CSV is read only once; reruns use the in-memory copy
def load_data():
    df = pd.read_csv("data/combined_data_hourly.csv", low_memory=False)
    df = df[df["Date"].between(MIN_DATE, MAX_DATE)]
    ## Dropping duplicates (daylight-saving "02X" hour)
    if "02X" in df["Hr_End"].values.astype(str):
        df.drop(df.loc[df["Hr_End"].astype(str)=="02X"].index, inplace=True)
    ## Converting HR_End variable to numeric to make Date_Time feature for time series
    df["Hr_End"] = df["Hr_End"].astype(int)
    return df

df = load_data()

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
                                    nbins=50,
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
    " time scale. To examine the design of the feature lag variables and differencing, we will use the " \
    "Augmented Dickey-Fuller test to check for trend stationarity and the OCSB test to check for seasonal "
    " differencing. Although, these test are not strictly necessary for random forrest, they do help us " \
    "decide the lag differencing in the model.")
    with st.container(border=True):
        st.subheader("Dickey-Fuller test to test for trends")
        st.latex(r"\mathbf{H_0} :  \mathrm{The\ time\ series\ is\ non-stationarity\ i.e\ }  \exists \mathrm{\ a\ unit\ root} ")
        st.latex(r"\mathbf{H_a} :  \mathrm{The\ time\ series\ is\ stationarity\ i.e\ }  \nexists \mathrm{\ a\ unit\ root} ")
        adf_result = adfuller(df["RT_Demand"], autolag="AIC")
        st.write(f"ADF Statistic: {adf_result[0]}")
        st.write(f"p-value: {adf_result[1]}")
        st.write("Assuming significance of 0.05, the p-value indicates that the data is stationary." \
        " Meaning that there are no trends in the data over the yearly time span.")

    with st.container(border=True):
        st.subheader("OCSB to check for seasonal differencing")
        st.write("The results indicate that trends are not present in the data, but our plots indicate a " \
        "seasonal pattern which may be important for determining our lags. We want to check whether" \
        " this seasonal pattern is stochastic or deterministic to see if seasonal differencing is needed for our model" \
        " or if  introducing a lag variable is sufficient. To address this, we will " \
        "consider an ocsb test to determine the lab periods for the features.")
        st.write("The following test indicates if the stated lag period is necessary")
        st.write(f"Daily lag period - {nsdiffs(df["RT_Demand"],m=24,test="ocsb")}")
        st.write(f"Weekly lag period - {nsdiffs(df["RT_Demand"],m=168 ,test="ocsb")}")
        st.write(f"Yearly lag period - {nsdiffs(df["RT_Demand"],m=8760,test="ocsb")}")
        st.write("Since the daily, weekly, and yearly periods return 0 on the ocsb test, we do not need " \
        "differencing. A lag should suffice to capture the periodic behavior in the data.")
    st.divider()
    # ── Two-Sample Independent t-test ─────────────────────────────────────────
    st.subheader("Bonus: t-test — Does Summer Demand Significantly Differ from Winter?")
    st.write(
        "Seasonal decomposition confirms a yearly cycle in RT_Demand. "
        "We use an independent two-sample t-test to statistically verify that "
        "summer (June–August) and winter (December–February) mean hourly demand are different."
    )
    with st.container(border=True):
        st.latex(r"\mathbf{H_0} : \mu_{\text{summer}} = \mu_{\text{winter}}")
        st.latex(r"\mathbf{H_a} : \mu_{\text{summer}} \neq \mu_{\text{winter}}")

        df_ttest = df.copy()
        df_ttest["Month"] = pd.to_datetime(df_ttest["Date"]).dt.month
        summer = df_ttest[df_ttest["Month"].isin([6, 7, 8])]["RT_Demand"]
        winter = df_ttest[df_ttest["Month"].isin([12, 1, 2])]["RT_Demand"]

        t_stat, p_value = stats.ttest_ind(summer, winter, equal_var=False)  # Welch's t-test
        alpha = 0.05

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Summer Mean (MW)", f"{summer.mean():,.1f}")
        col_b.metric("Winter Mean (MW)", f"{winter.mean():,.1f}")
        col_c.metric("Difference (MW)", f"{summer.mean() - winter.mean():,.1f}")

        st.write(f"**t-statistic:** {t_stat:.4f}")
        st.write(f"**p-value:** {p_value:.2e}")

        if p_value < alpha:
            st.success(
                f"p-value ({p_value:.2e}) < α (0.05) → **Reject H₀**. "
                "There is a statistically significant difference in mean hourly RT_Demand "
                "between summer and winter. Summer demand is on average "
                f"{summer.mean() - winter.mean():,.0f} MW higher, driven by air-conditioning load."
            )
        else:
            st.info("p-value ≥ α (0.05) → Fail to reject H₀.")

        # Visual comparison
        import plotly.graph_objects as go
        fig_ttest = go.Figure()
        fig_ttest.add_trace(go.Histogram(x=summer, name="Summer (Jun–Aug)", opacity=0.65, nbinsx=60))
        fig_ttest.add_trace(go.Histogram(x=winter, name="Winter (Dec–Feb)", opacity=0.65, nbinsx=60))
        fig_ttest.update_layout(
            barmode="overlay",
            title="RT_Demand Distribution: Summer vs. Winter",
            xaxis_title="RT_Demand (MW)",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_ttest, use_container_width=True)

with tab5:  # ML forecast
    st.subheader("Random Forest Forecast")
    st.write(
        "We create a 24-hour, 168-hour (weekly), and use time-series cross validation on the training set." \
        " Addtionally, a Month feature is added to capture yearly fluctuations. "\
        "Finally, choose the parameter combination with the lowest average MAE."
    )
    lags =[24, 48, 168]
    df_lagged = df.copy()
    # Build an hourly timestamp so the rows can be ordered correctly before creating lags.
    # This is important because shift(24) means "24 rows earlier", so the dataframe must
    # be in true time order for the lag to represent the same hour on the previous day.
    df_lagged["DateTime"] = pd.to_datetime(df_lagged["Date"]) + pd.to_timedelta(
        df_lagged["Hr_End"] - 1, unit="h"
    )
    df_lagged = df_lagged.sort_values("DateTime")

    # Use demand from the same hour one day earlier as the predictor.
    for lag in lags:
        df_lagged[f"RT_Demand-{lag}"] = df_lagged["RT_Demand"].shift(lag)
   
    # Adding month for yearly periodicity
    df_lagged["Month"] = df_lagged["DateTime"].dt.month
    # The first  168 rows have no previous-day value, so we remove them.
    df_lagged = df_lagged.dropna()
    # Train on data before 2025 and test on 2025 onward.
#    Instead of doing the hold-out 80/20 method, we are going to estimate out of sample error by
#    random sampling 2 24 hour periods for every week in the last year. The idea is to get a sampling 
#    distribution of the out of sample error on the most recent data. To prevent leaking, we will retain
#    that out of sample folds are on later data than training. 

#    The rationale for this method is that--I think, and this "think" should be heavily emphasized-- this
#    method is more representative of the use case. The problem from industry that we are simulating
#    is predicting how much energy to purchace for tomorrow. A company like National grid puts in a bid
#    at 10am today to purchase XXXX MW of energy for a given region for each HR in the next day. 

#    Our original method of testing on the last year has the model predict too far away. Based on the
#    toy problem, the model needs to predict 24 hours in advance so testing its ability to forecast next
#    year does not capture the performance we want. Sampling 24 hours from the last year, however,
#    does give us a sense to how the model performs while accounting for the yearly and weekly variation
#    we noticed when examining the seasonal patterns in the data.  
   
#    I could be making a mistake here, I dont fully know if the previous method is bad. I implemented 
#    SARIMA for the same problem using the previous and found the model smoothed the data too 
#    much when fitting the model on the yearly scale. RF may not suffer from the same problem due 
#    to its structure. But, for the sake of consistency, I want to use the same evaluation scheme. 
   
#    For the purposes of this project, this is not relevant. But I do think the extra effort should be 
#    instructive for turning this project into a Capstone, hence, the choice to make this evaluation setup. 
#    IDK, learning here.  
    # This keeps the split chronological, which is required for time-series modeling.
    training_data = df_lagged[df_lagged["Date"] < "2025-01-01"].copy()
    test_data = df_lagged[df_lagged["Date"] >= "2025-01-01"].copy()
    seed = 43 # I am a fan of primes 

   # DA_Demand is a response variable that is highly correlated with RT_Demand
   # Domain knowledge - DA_Demands is the estimate of what RT_Demand will be  
    X_train_full = training_data.drop(columns=["RT_Demand","DA_Demand","Date","DateTime"])
    y_train_full = training_data["RT_Demand"]
    X_test = test_data.drop(columns=["RT_Demand","DA_Demand","Date","DateTime"])
    y_test = test_data["RT_Demand"]

    selector = SelectFromModel(RandomForestRegressor(n_estimators=200, random_state= seed),
                               threshold="mean")
    selector.fit(X_train_full, y_train_full) 

    importances = selector.estimator_.feature_importances_
    feature_importance_df = pd.DataFrame({
    'feature': X_train_full.columns,
    'importance': importances,
    'selected': selector.get_support()
    }).sort_values('importance', ascending=False)

    st.dataframe(feature_importance_df, use_container_width=True)
    X_train_selected = selector.transform(X_train_full)
    X_test_selected = selector.transform(X_test)

    st.markdown("**Training/Test Split**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Training rows", f"{len(training_data):,}")
    col2.metric("Test rows", f"{len(test_data):,}")
    col3.metric("Lag used", f"{lag} hours")

    # Parameter grid to try during cross validation.
    n_splits = st.slider("Number of CV folds", min_value=3, max_value=8, value=5, step=1)
    param_grid = {
       "n_estimators":[10,50,100, 200],
       "max_depth":[None,10,20,30],
       'min_samples_split': [2, 5, 10],
       'min_samples_leaf': [1, 2, 4],
       } 
    # chose these by copying this medium article. Honestly, don't yet know why. This is another thing
    # to learn. 
    #LINK: https://medium.com/@Doug-Creates/tuning-random-forest-parameters-with-scikit-learn-b53cbc602cd0
    
    search = GridSearchCV(
        RandomForestRegressor(random_state=seed),
        param_grid,
        cv = TimeSeriesSplit(n_splits=n_splits),
        scoring=["neg_root_mean_squared_error",
                        "neg_mean_absolute_percentage_error",
                        "neg_mean_absolute_error"],
        refit="neg_root_mean_squared_error",
        n_jobs=-1 # lot of jobs, lots of compute
    )

    # Chose scoring based on what we use later for reporting. RMSE gives us average in units of MW
    # MAPE is nice because we can get a sense of scale. 4% increase in accuracy could mean lots of 
    # money so if we were to report this to decision makers on the team, that would help a lot.
    
    search.fit(X_train_selected, y_train_full)
    # # TimeSeriesSplit preserves time order:
    # # earlier data is used for training, later data for validation.
    # tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = pd.DataFrame(search.cv_results_) 
    summary_df = cv_results[[
        "param_n_estimators","param_max_depth",
        "param_min_samples_split","param_min_samples_leaf",
        "mean_test_neg_mean_absolute_error",
        "mean_test_neg_root_mean_squared_error",
        "mean_test_neg_mean_absolute_percentage_error"
    ]].copy()

    summary_df.columns = [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "avg_mae",
        "avg_rmse",
        "avg_mape"
        ]
    # transforming report values to positive readable format 
    summary_df[["avg_mae","avg_rmse","avg_mape"]] *= -1
    summary_df["avg_mape"] *= 100 # now as a percent
    summary_df.sort_values("avg_mae")
    # best_mae = float("inf")
    # best_params = {}

    # # Try every parameter combination and compute validation MAE on each fold.
    # for n_estimators in n_estimators_list:
    #     for max_depth in max_depth_list:
    #         fold_maes = []
    #         fold_rmses = []
    #         fold_mapes = []

    #         for fold_number, (train_index, val_index) in enumerate(tscv.split(X_train_full), start=1):
    #             X_train = X_train_full.iloc[train_index]
    #             X_val = X_train_full.iloc[val_index]
    #             y_train = y_train_full.iloc[train_index]
    #             y_val = y_train_full.iloc[val_index]

    #             model = RandomForestRegressor(
    #                 n_estimators=n_estimators,
    #                 max_depth=max_depth,
    #                 random_state=42,
    #                 n_jobs=-1,
    #             )
    #             model.fit(X_train, y_train)
    #             y_pred = model.predict(X_val)
    #             fold_mae = mean_absolute_error(y_val, y_pred)
    #             fold_rmse = sqrt(mean_squared_error(y_val, y_pred))
    #             fold_mape = mean_absolute_percentage_error(y_val, y_pred) * 100
    #             fold_maes.append(fold_mae)
    #             fold_rmses.append(fold_rmse)
    #             fold_mapes.append(fold_mape)

    #             # Save each fold result so it can be shown later in Streamlit.
    #             cv_results.append(
    #                 {
    #                     "n_estimators": n_estimators,
    #                     "max_depth": "None" if max_depth is None else max_depth,
    #                     "fold": fold_number,
    #                     "mae": fold_mae,
    #                     "rmse": fold_rmse,
    #                     "mape": fold_mape,
    #                 }
    #             )

    #         # Average the fold errors for this parameter combination.
    #         avg_mae = sum(fold_maes) / len(fold_maes)
    #         if avg_mae < best_mae:
    #             best_mae = avg_mae
    #             best_params = {"n_estimators": n_estimators, "max_depth": max_depth}

    # cv_results_df = pd.DataFrame(cv_results)
    # # Summarize one row per parameter combination for the table and chart.
    # summary_df = (
    #     cv_results_df.groupby(["n_estimators", "max_depth"], as_index=False)[["mae", "rmse", "mape"]]
    #     .mean()
    #     .rename(columns={"mae": "avg_mae", "rmse": "avg_rmse", "mape": "avg_mape"})
    #     .sort_values("avg_mae")
    # )
    # best_summary_row = summary_df.iloc[0]

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
    st.write(search.best_params_)

    # st.write(
    #     {
    #         "n_estimators": best_params["n_estimators"],
    #         "max_depth": "None" if best_params["max_depth"] is None else best_params["max_depth"],
    #         "cv_mae": round(best_summary_row["avg_mae"], 2),
    #         "cv_rmse": round(best_summary_row["avg_rmse"], 2),
    #         "cv_mape_percent": round(best_summary_row["avg_mape"], 2),
    #     }
    # )
    best_summary_row = summary_df.iloc[0]
    st.markdown("**Validation Performance (Average Across Folds)**")
    val_col1, val_col2, val_col3 = st.columns(3)
    val_col1.metric("Validation MAE", f"{best_summary_row['avg_mae']:,.2f}")
    val_col2.metric("Validation RMSE", f"{best_summary_row['avg_rmse']:,.2f}")
    val_col3.metric("Validation MAPE", f"{best_summary_row['avg_mape']:,.2f}%")

    # # Refit the model on the full training set using the winning parameters.
    # final_model = RandomForestRegressor(
    #     n_estimators=best_params["n_estimators"],
    #     max_depth=best_params["max_depth"],
    #     random_state=42,
    #     n_jobs=-1,
    # )

    # # Refit the model on the full training set using the winning parameters.
    final_model = search.best_estimator_
    # final_model.fit(X_train_full, y_train_full)
    test_predictions = final_model.predict(X_test_selected)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_rmse = sqrt(mean_squared_error(y_test, test_predictions))
    test_mape = mean_absolute_percentage_error(y_test, test_predictions) * 100

    # # Store actual and predicted values together for plotting.
    predictions_df = test_data[["DateTime", "RT_Demand"]].copy()
    predictions_df["Predicted_RT_Demand"] = test_predictions

    st.markdown("**Test Set Performance**")
    test_col1, test_col2, test_col3 = st.columns(3)
    test_col1.metric("Test MAE", f"{test_mae:,.2f}")
    test_col2.metric("Test RMSE", f"{test_rmse:,.2f}")
    test_col3.metric("Test MAPE", f"{test_mape:,.2f}%")

    # # Show only the first 14 test days so the chart stays readable.
    forecast_chart = px.line(
        predictions_df.head(24 * 14),
        x="DateTime",
        y=["RT_Demand", "Predicted_RT_Demand"],
        title="Actual vs Predicted RT_Demand (First 14 Days of Test Set)",
        labels={"value": "RT_Demand", "variable": "Series"},
    )
    st.plotly_chart(forecast_chart, use_container_width=True)

    # # Interpretation of the Results
    st.divider()
    st.subheader("Interpretation of the Results")
    st.write(
        "Overall we are pretty satisfied with how the model performed. Selected features: "
        "the RT_Demand from the same hour the previous day (24-hour lag),  and Dry Bulb temperature"
        ". This makes sense given what is typically expected from models in the domain. A regression" \
        " study on the same data that was done last semester revealed that Heating Degree Days which " \
        " is an aggregate indicator of temperature was the most significant predictor for Total demand " \
        " thus, the selection makes aligns with earlier results in the model. Further, since the data response " \
        " is expected to have high correlation with the time before--evident in the key findings, the importance" \
        " of the lag 24 is reasonable as well. The model was trained on 2018 through 2024 and evaluated on the fully "
        "held-out 2025 to simulate the business use for this kind of model."
    )
    with st.container(border=True):
        st.markdown("**What the metrics are actually telling us**")
        st.write(
            "MAE (Mean Absolute Error) is probably the easiest metric to interpret here: it tells "
            "us how far off the model is on average in megawatts. Since average RT_Demand hovers "
            "around 2,700 MW, an MAE in the 80 to 130 MW range means the model is typically within "
            "about 3 to 5 percent of the actual value, which is what the MAPE confirms. "
            "RMSE penalizes larger errors more heavily than MAE does, so when RMSE is close to MAE "
            "it means the model is not making a lot of very bad predictions on specific hours. "
            "The thing we paid most attention to is that the cross-validation MAE and the test MAE "
            "are in the same ballpark. If the test error had been much worse than CV, that would "
            "have been a sign of overfitting to the training years, but that does not appear to "
            "be happening here. In fact, we are doing slightly better on out of sample performance indicating" \
            " the model selection process may be able to generalize well. To examine this we would have to nest " \
            "the feature selection and hyperparameter searching in an outer CV to get a sampling distribution " \
            "for the modeling process. So we cannot overclaim performance here, despite positive outlook from this test."
        )
    with st.container(border=True):
        st.markdown("**How this connects back to the hypothesis tests and EDA**")
        st.write(
            "The stationarity tests (ADF) and seasonality test (OCSB) both pointed to the same conclusion: RT_Demand "
            "does not have a long-run trend nor a stochastic seasonality pattern," \
            " so we did not need to difference the series before modeling." \
            " That is part of why lag-based features work as well as they do. The t-test "
            "showed that summer demand is about 270 MW higher than winter demand on average, and "
            "the Month feature directly captures this by telling the model what time of year it is. "
            "The 24-hour lag handles the daily pattern and the 168-hour lag picks up the weekly "
            "rhythm we saw in the Key Findings tab. It is interesting that the model did not consider these " \
            "to be important, this may have to do with the way the model is being tested. Since it is forward " \
            "24 hours ahead, the correlation in the demand may capture the higher level fluctuations, or " \
            "the feature selection setup is not capturing that behavior. Additionaly, there could be an explanation " \
            "that we have not considered here."
        )
    with st.container(border=True):
        st.markdown("**Where the model still struggles**")
        st.write(
            "Even with these features, the model has no way to anticipate sudden demand deviations "
            "caused by extreme weather events or major holidays. If last week and yesterday were "
            "both normal days but today deviates greatly from the estimated inputs, the model may not "
            "capture that that variation. Although, literature from the domain indicates that tree based " \
            "and deep learning methods generalize quite well [1,2,3,4,5] You can see this in the forecast " \
            "chart when the predicted line and the actual line diverge the most. Those tend to be the " \
            "most operationally important days from a grid perspective, which is exactly when you " \
            "need the forecast to be reliable."
        )

    # Improvements and Next Steps
    st.divider()
    st.subheader("Improvements and Next Steps")
    col_imp1, col_imp2 = st.columns(2)
    with col_imp1:
        st.markdown("**Features we would add next**")
        st.write(
            "We would also add a day-of-week feature and a public holiday flag, since the weekly " \
            "pattern in the Key Findings tab shows that weekends are consistently lower than weekdays "
            "and the current model has no way to know what day it is."
        )
    with col_imp2:
        st.markdown("**Modeling improvements worth trying**")
        st.write(
            "With a richer feature set, it would make sense to try XGBoost or LightGBM. They "
            "tend to train faster than Random Forest and often do better on tabular data once the "
            "feature set grows. For a more ambitious version of this project, an LSTM network "
            "could learn the daily and yearly seasonality directly from the sequence without "
            "needing us to engineer lag features manually. We would also want to expand the "
            "hyperparameter search a bit. Finally, adding prediction intervals to the output would make " \
            "the forecasts more useful in practice, since knowing the uncertainty around a prediction " \
            "matters just as much as the prediction itself when you are managing a power grid."
        )