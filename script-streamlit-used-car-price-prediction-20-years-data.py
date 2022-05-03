# %% [markdown]
# # Used Car Price Prediction - 20 Years Data
# Vinod Shivarudrappa
# vshiv@bu.edu

# %%
# Numpy & Pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp

# Models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn import metrics
import missingno as msno

# Plotting
from warnings import simplefilter
import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px

# GCP Auth
from google.oauth2 import service_account
from google.cloud import storage

# Others
import io

# %% [markdown]
st.title("Used Car Price Prediction - 20 Years Data")

st.write("Using the last 20 years' worth of data, the below algorithms are to best predict the value of used cars, based on 10 features;")
st.write("* Linear Regression\n* Decision Trees\n* Bagging\n* Random Forest\
        \n* Adaptive Boosting\n* Gradient Boosting\n* XGBoost")

# %%


def read_data_from_gcp(bucket, file_path):
    """ Retrieve the file contents into a dataframe and return it """

    try:
        vehicles_df_full = pd.read_csv(bucket.blob(
            file_path).open(), header=0, sep=',', encoding='utf-8')

    except Exception as e:
        st.error(f"Error reading file: {e}")

    else:
        return vehicles_df_full


# %%
if __name__ == '__main__':

    # Declare variables required
    DASHES = '-' * 10
    TABS = '\t' * 8

    BUCKET_NAME = "used-car-dataset"
    FILE_NAME = "vehicles.csv" #"vehicles_reduced.csv"

    ########### Step 1 - Read Data ############
    st.write(DASHES)
    st.subheader(f"Step 1 - Read Data from the cloud")

    # Authentication to GCP
    # Create API client
    credentials = service_account.Credentials.\
        from_service_account_info(st.secrets["gcp_service_account_vshiv_svc1"])
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(BUCKET_NAME)

    # Read data from the file into a dataframe and display the top 10 records

    with st.spinner(f'Reading input file {FILE_NAME}...'):
        vehicles_df_full = read_data_from_gcp(bucket, FILE_NAME)
        #vehicles_df_full = pd.read_csv(FILE_NAME, header=0) # testing line
        vehicle_count = len(vehicles_df_full)
        if vehicle_count > 0:
            st.success(
                f"Read {len(vehicles_df_full)} lines from {FILE_NAME}\n")
            st.write(f"Here are the top 10 records:")
            st.dataframe(data=vehicles_df_full.head(10))

# %%
    ########### Step 2 - Assess Completeness of Data ############
    st.write(DASHES)
    st.subheader("Step 2 - Assess Completeness of Data")

    st.write("By doing a simple count operation on the dataframe and plotting it as a bar chart, the completeness of data becomes apparent:")

    fig = px.bar(vehicles_df_full.count(), text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

    st.write("With that, the fields that are to be cleaned up become apparent.")
    st.write("Some fields don't add value to this exercise as well. Removing them, the resultant dataframe is as below:")

    # Determine and remove the columns to drop based on the above graph
    cols_to_drop = ['id', 'url', 'region', 'region_url', 'VIN', 'image_url', 'description',
                    'county', 'size', 'paint_color', 'drive', 'cylinders', 'state', 'lat', 'long']
    vehicles_df = vehicles_df_full.drop(columns=cols_to_drop)

    # Remove the larger data frame from memory
    del vehicles_df_full

    # Initial cleaning up
    # Drop NaNs and duplicates
    vehicles_df.dropna(inplace=True)
    vehicles_df.drop_duplicates(inplace=True)

    # Update index and change data type of year to string
    vehicles_df.index = range(len(vehicles_df))
    vehicles_df.year = vehicles_df.year.astype(int).astype(str)

    # Plot the cleaned up DF
    fig = px.bar(vehicles_df.count(), text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# %%

    ########### Step 3 - Data Visualization and Cleaning ############
    st.write(DASHES)
    st.subheader("Step 3 - Data Visualization and Cleaning")

    st.write("Visualizing the data reveals patterns that are not obvious to the human eye when reviewing raw data.\nCorrelation matrices, histograms, category, scatter & box plots have helped identify relationships")
    st.write("Cleaned data based on visualizations:\n* Removed NaNs & duplicates\n* Price b/w 2k and 50k\n* Odometer b/w 100 and 200k, etc..")

    st.write(
        "In the section below, features that would help with better prediction are identified")

    st.write("First step is to describe the cleaned up dataframe to understand the non-categorical features as below:")
    st.dataframe(vehicles_df.describe())

    # Ananlyze PRICE
    st.markdown('\n##### Analyzing PRICE')
    st.write("It appears that the target field - PRICE ranges between 0 and an unrealistic $3.7B. Plotting a distribution of the same:")
    fig = px.histogram(data_frame=vehicles_df, x='price',
                       histfunc='count', nbins=5, text_auto=True, facet_col_wrap=1)
    fig.update_traces(textfont_size=12, textangle=0,
                      textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)

    st.write(
        "\nTo keep things simple and realistic, making a subset of prices between 2k and 50k:")
    vehicles_prc = vehicles_df[(vehicles_df.price >= 2000) & (
        vehicles_df.price <= 50000)]
    fig = px.histogram(data_frame=vehicles_prc, x='price',
                       histfunc='count', nbins=5, text_auto=True, facet_col_wrap=1)
    fig.update_traces(textfont_size=12, textangle=0,
                      textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)

    # Ananlyze ODOMETER
    st.markdown('\n##### Analyzing ODOMETER')
    st.write(
        f"Assessing the skewness for odometer readings, it was found to be: {round(vehicles_prc['odometer'].skew(),2)}")

    st.write(f"Distributing the same on a violin chart:")
    fig = px.violin(data_frame=vehicles_df, x='odometer', points=False,
                    violinmode='group', box=True)
    st.plotly_chart(fig, use_container_width=True)

# %%

# %%
