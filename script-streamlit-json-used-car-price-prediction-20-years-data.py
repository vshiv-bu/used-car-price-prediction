# %% [markdown]
# # Used Car Price Prediction - 20 Years Data
# Vinod Shivarudrappa
# vshiv@bu.edu

# %%
# Numpy & Pandas
import pandas as pd
import numpy as np


# Models
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
# import xgboost as xgb
# from sklearn import metrics
# import missingno as msno

# Streamlit
import streamlit as st
import plotly.express as px
from annotated_text import annotated_text, annotation

# GCP Auth
from google.oauth2 import service_account
from google.cloud import storage

# %% [markdown]
st.title("Used Car Price Prediction - 20 Years Data")

st.write("Using the last 20 years' worth of data, the below algorithms are to best predict the value of used cars, based on 10 features;")
st.write("* Linear Regression\n* Decision Trees\n* Bagging\n* Random Forest\
        \n* Adaptive Boosting\n* Gradient Boosting\n* XGBoost")

# %%


def read_data_from_gcp(bucket, file_path):
    """ Retrieve the file contents into a dataframe and return it """

    try:
        vehicles_df_full = pd.read_json(bucket.blob(
            file_path).open())

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
    FILE_NAME = "vehicles.json"  # "vehicles_reduced.csv"

    ########### Step 1 - Read Data ############
    st.write(DASHES)
    st.subheader(f"Step 1 - Read data from the cloud")

    # Authentication to GCP
    # Create API client
    credentials = service_account.Credentials.\
        from_service_account_info(st.secrets["gcp_service_account_vshiv_svc1"])
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(BUCKET_NAME)

    # Read data from the file into a dataframe and display the top 10 records

    with st.spinner(f'Reading input file {FILE_NAME}...'):
        vehicles_df_full = read_data_from_gcp(bucket, FILE_NAME)
        #vehicles_df_full = pd.read_json(FILE_NAME)  # testing line
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
    cols_to_drop = ['region', 'VIN', 'county', 'size',
                    'paint_color', 'drive', 'cylinders', 'state', 'image_url']
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

    # Analyze PRICE
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

    # Analyze ODOMETER
    st.markdown('\n##### Analyzing ODOMETER')

    annotated_text(
        ("Assessing the skewness for odometer readings, it was found to be:"),
        annotation(str(round(vehicles_prc['odometer'].skew(), 2)), color="#8ef"))

    st.write(f"Distributing the same on a violin chart:")
    fig = px.violin(data_frame=vehicles_prc, x='odometer', points=False,
                    violinmode='group', box=True)
    st.plotly_chart(fig, use_container_width=True)

    st.write("It's evident that the distribution is highly skewed and there's some bad data with max odometer readings of 10mil miles etc.")

    st.write(
        "Research, finds that Americans drive an average of 14,300 miles per year, according to the [Federal Highway Administration](https://www.thezebra.com/resources/driving/average-miles-driven-per-year/).")

    st.write("Analyzing the distribution where.")

    with st.container() as c:
        'odometer = 0:'
        st.dataframe(vehicles_prc[(vehicles_prc.odometer == 0)].describe())
        'odometer > 200k:'
        st.dataframe(vehicles_prc[(vehicles_prc.odometer > 200000)].describe())

    st.write("Based on the stats above, it's fair to continue with a dataset filtered for odometer readings be between 100 (CPO) to 200k (20 yo).")

    vehicles_odo = vehicles_prc[(vehicles_prc.odometer > 100)
                                & (vehicles_prc.odometer <= 200000)]
    annotated_text(
        ("With that, the skewness becomes: "),
        annotation(str(round(vehicles_odo['odometer'].skew(), 2)), color="#afa"))

    st.write("Here's the distribution of the same:")
    fig = px.histogram(data_frame=vehicles_odo, x='odometer',
                       histfunc='count', nbins=5, text_auto=True, facet_col_wrap=1)
    fig.update_traces(textfont_size=12, textangle=0,
                      textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)

    # Analyze categorical relationships
    st.markdown('\n##### Analyzing Categorical Relationships')

    # Price vs. Year
    st.markdown('\n###### **Price vs. Year**')
    st.write("Plotting the relationship of Price vs Year of Manufacture - it's evident that the data is not consistent:")
    fig = px.box(data_frame=vehicles_odo,
                 x=vehicles_odo.year.astype(int), y='price')
    st.plotly_chart(fig)

    st.write("Filtering this dataset to include data between 2000 and 2020:")
    year_list = list(range(2000, 2021))
    vehicles_year = vehicles_odo[vehicles_odo.year.astype(int).isin(year_list)]
    fig = px.box(data_frame=vehicles_year,
                 x=vehicles_year.year.astype(int), y='price')
    st.plotly_chart(fig)

    st.write("That's better. With this used 20 year set, next, trying to find how the three features come together and depict real-world characteristics..")

    # Price vs. Mean Odometer Readings
    st.markdown(
        "\n###### Price vs Mean Odometer readings over the age of the car posted")
    st.write("Adding a computed field called 'age' to the dataframe:")
    # Calculate age of the posted car using "posting date"
    # Convert year and posting date to datetime
    vehicles_year.posting_date = pd.to_datetime(
        vehicles_year.posting_date, utc=True)
    vehicles_year.posting_date = vehicles_year.posting_date.astype(
        'datetime64[ns]')

    # Add a new field for age of cars
    vehicles_year['age'] = vehicles_year.posting_date.dt.year.astype(
        int) - vehicles_year.year.astype(int)

    # Get a preview of the changes
    st.dataframe(vehicles_year.head(5))

    st.write("and plotting a scatter of Age vs Price vs Mean Odometer Readings:")
    grp_df = vehicles_year.groupby(by='age').mean()[['price','odometer']].astype(int).reset_index()

    # Visualize how odometer average readings vary with price over age of cars
    # Set axes and points 
    x = grp_df.price
    y = grp_df.odometer
    z = grp_df.age
    size = [50*n for n in range(len(y))]

    fig = px.scatter_3d(x=x, y=y, z=z, size=size, opacity=1, \
            labels= {'x': 'price', 'y':'odometer', 'z':'age'})
    st.plotly_chart(fig)

    st.write("It's evident from the visualization above that cars that have been driven less are more expensive than older cars which have been driven more. There seem to be a good chunk of cars under 10k that have been driven 120k and over and are 12 years and older - this is an interesting insight")



# %%

# pr = vehicles_df.profile_report()
# st_profile_report(pr)
# %%
