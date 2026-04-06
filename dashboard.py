import pandas as pd  # Pandas for data manipulation
import streamlit as st  # Streamlit library for web apps (dashboards for our data)
import plotly.express as px  # Plotly Express for creating charts

st.title("Final Project")  # Title of the dashboard
st.header("Ramson Munoz & Valentina Kloster")  # Subtitle of the dashboard
st.subheader("Forcasting real time energy demand")  # Header of the dashboard

# To run: streamlit run dashboard.py

st.divider()

st.sidebar.header("Load Datasets")
file_uploaded = st.sidebar.file_uploader("Upload a file", type=["csv"])
if file_uploaded is not None:
    df = pd.read_csv(file_uploaded)
else:
    df = pd.read_csv("data/combined_data_hourly.csv")
    # we just want to include the records from 2018 to 2025
    df = df[df["Date"].between("2018-01-01", "2025-12-31")]

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


#     st.subheader("Charts")
#     with st.expander("Scatter Plot"):
#         fig1 = px.scatter(df,
#                           x="Total Water Column (m)",
#                           y="Temperature (c)",
#                           color="Salinity (ppt)",
#                           title="Temperature vs Total Water Column",)
#         st.plotly_chart(fig1) # you pass the chart we created using plotly express
#     with st.expander("Line Chart"):
#         col1, col2 = st.columns([1,4]) # two columns, first one is 1/5 of the width, second one is 4/5 of the width
#         with col1:
#             parameter_selected = st.selectbox("Select Parameter",
#                                               df.columns) # user selection
#             color_selected = st.color_picker("Select a Color",
#                                               "#6495ED") # user color selection, with defualt #6495ED
#         with col2:
#             fig2 = px.line(df,
#                         x="Time",
#                         y=parameter_selected, # user selection
#                         title=f"{parameter_selected} Over Time",
#                         color_discrete_sequence=[color_selected]) # user color selection
#             st.plotly_chart(fig2)
#     with st.expander("3D Scatter Plot"):
#         # BEcause I have lat, lon, depth, I can create a 3D scatter plot
#         fig3 = px.scatter_3d(df,
#                              x="Longitude",
#                              y="Latitude",
#                              z="Total Water Column (m)",
#                              color="ODO mg/L",
#                              title="3D Scatter Plot of Water Depth")
#         fig3.update_scenes(zaxis_autorange='reversed') # Reverse the z-axis to have depth going downwards (because is the water depth)
#         st.plotly_chart(fig3)
# with st.expander("Box Plot"):
#    numeric_columns = df.select_dtypes(include="number").columns
#    feature_selector = st.selectbox("Select a parameter to visualize", numeric_columns)
#    fig5 = px.box(df, x=f"{feature_selector}", title=f"Box Plot of {feature_selector}")
#    st.plotly_chart(fig5)

#     with st.expander("Histogram"):
#         col1, col2 = st.columns([1,4]) # two columns, first one is 1/5 of the width, second one is 4/5 of the width
#         with col1:
#             parameter_selected2 = st.selectbox("Select Parameter 2", df.columns) # user selection
#         with col2:
#             fig6 = px.histogram(df,
#                                 x=parameter_selected2,
#                                 nbins=30)
#             st.plotly_chart(fig6)
#     with st.expander("Heatmap"):
#         df_numeric = df.select_dtypes(include=['number']) # select only numeric columns
#         fig7 = px.imshow(df_numeric.corr(),
#                         text_auto=True,
#                         title="Heatmap of Correlation Matrix")
#         st.plotly_chart(fig7)

#with tab3:
#     st.subheader("Maps")
#     fig4 = px.scatter_mapbox(df,
#                              lon="Longitude", # this is our x axis
#                              lat="Latitude", # this is our y axis
#                              color="Temperature (c)",
#                              hover_data=df,
#                              mapbox_style="open-street-map", # style of the map (open-street-map is free)
#                              zoom=17, # initial zoom level
#                              )
#     fig4.update_layout(margin={"r":0,"t":0,"l":0,"b":0}) # remove margins
#     st.plotly_chart(fig4)
#     st.caption("Hover over the points to see more details")

# # TODO: add more charts, such as
# # box plots, histograms, and heatmaps for correlation (as seen in last week's class)

#  with tab5: # ML forecast
## Feature engineering
### TODO introduce lags for random forrest

### TODO Cross validation for param selection in the random forrest setup

### TODO Plot of per fold metrics side by side column chart of validation error per fold vis RMSE, MAPE

### TODO FINAL CHART Test RMSE MAPE
