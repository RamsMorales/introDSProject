import pandas as pd # Pandas for data manipulation
import streamlit as st # Streamlit library for web apps (dashboards for our data)
import plotly.express as px  # Plotly Express for creating charts

st.title("Final Project")  # Title of the dashboard
st.header("Ramson Munoz & Valentina Kloster")  # Subtitle of the dashboard
st.subheader("Forcasting real time energy demand")  # Header of the dashboard

# To run: streamlit run dashboard.py

st.divider()

st.sidebar.header("Load Datasets")
file_uploaded = st.sidebar.file_uploader("Upload a file", type=["csv"]) # File uploader in the sidebar (because we used st.sidebar.file_uploader its gonna be inside the sidebar)
if file_uploaded is not None: # If a file is uploaded
    df = pd.read_csv(file_uploaded) # Read the uploaded CSV file
else:
    df = pd.read_csv("data/combined_data_hourly.csv")  # Default dataset

# tab1, tab2, tab3 = st.tabs(["Raw Data", "Charts", "Maps"])  # Creating three tabs

# with tab1:
#     st.subheader("Raw Data")
#     st.dataframe(df)
#     st.caption("Data collected from an underwater robot by Dr. Greg Reis")

#     st.divider()

#     st.subheader("Summary Statistics")
#     st.dataframe(df.describe())

# with tab2:
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
#     with st.expander("Box Plot"):
#         fig5 = px.box(df,
#                       y="Salinity (ppt)",
#                       title="Box Plot of Salinity")
#         st.plotly_chart(fig5)
        
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

# with tab3:
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

