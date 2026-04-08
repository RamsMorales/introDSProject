import pandas as pd  # Pandas for data manipulation
import streamlit as st  # Streamlit library for web apps (dashboards for our data)
import plotly.express as px  # Plotly Express for creating charts
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
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


#with tab5: # ML forecast
## Feature engineering
df_lagged = df.copy()
lag = 24 #hours
df_lagged["RT_Demand-24"] = df_lagged["RT_Demand"].shift(lag)
df_lagged= df_lagged.dropna()
# Train test splitting. Testing on the last year of observations of RT_Demand
test_data = df_lagged[df_lagged["Date"] > "2023-12-31"]
training_data = df_lagged[df_lagged["Date"] < "2024-01-01"]

### TODO Cross validation for param selection in the random forrest setup

### TODO Plot of per fold metrics side by side column chart of validation error per fold vis RMSE, MAPE

### TODO FINAL CHART Test RMSE MAPE
