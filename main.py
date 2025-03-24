import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
import os
from scripts import download, data_cleaning 

# Set up BigQuery credentials
project_id = "brave-drummer-454602-n7"
dataset_id = "ecommercedataset"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\HP\Downloads\brave-drummer-454602-n7-0ec1f7058e09.json"
client = bigquery.Client(project=project_id)

# Function to load dataset
def load_dataset(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Function to visualize data
def visualize_data(df, dataset_name):
    st.write(f"### {dataset_name} Dataset Preview")
    st.write(df.head())
    
    st.write("### Missing Values")
    st.write(df.isna().sum())
    
    st.write("### Data Distribution & Visualization")
    continuous_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in continuous_cols:
        st.write(f"#### Distribution of {col}")
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)
        
        st.write(f"#### Boxplot of {col}")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)

# Function to fetch BigQuery data
def fetch_bigquery_data(query):
    try:
        query_job = client.query(query)
        results = query_job.result()
        return results.to_dataframe()
    except Exception as e:
        st.error(f"BigQuery Error: {e}")
        return None

# Streamlit App
def main():
    st.title("E-commerce Data Processing & Visualization")

    if st.button("Download Data"):
        st.write("Downloading data...")
        download.download()
        st.success("Data downloaded successfully!")

    # Buttons for executing scripts
    if st.button("Clean Data"):
        st.write("Cleaning data...")
        data_cleaning.data_cleaning()
        st.success("Data cleaning completed!")

    if st.button("Generate Synthetic Events"):
        st.write("Generating synthetic e-commerce events...")
        data_cleaning.data_cleaningEvent()
        st.success("Synthetic events generated successfully!")

    # Sidebar for dataset selection
    st.sidebar.title("Dataset Selection")
    dataset_option = st.sidebar.selectbox(
        "Choose a dataset to visualize:", 
        ["None", "Customers", "Orders", "Products", "Payments", "Reviews", "Sellers"]
    )
    
    dataset_paths = {
        "Customers": "c:/Users/HP/Downloads/E-commerce pipeline cleaned data/olist_customers_cleaned_dataset.csv",
        "Orders": "c:/Users/HP/Downloads/E-commerce pipeline cleaned data/olist_orders_cleaned_dataset.csv",
        "Products": "c:/Users/HP/Downloads/E-commerce pipeline cleaned data/olist_products_cleaned_dataset.csv",
        "Payments": "c:/Users/HP/Downloads/E-commerce pipeline cleaned data/olist_order_payments_cleaned_dataset.csv",
        "Reviews": "c:/Users/HP/Downloads/E-commerce pipeline cleaned data/olist_order_reviews_cleaned_dataset.csv",
        "Sellers": "c:/Users/HP/Downloads/E-commerce pipeline cleaned data/olist_sellers_cleaned_dataset.csv"
    }

    # Load and display selected dataset
    if dataset_option != "None":
        df = load_dataset(dataset_paths[dataset_option])
        if df is not None:
            visualize_data(df, dataset_option)

    # BigQuery Queries Section
    st.sidebar.title("BigQuery Analytics")
    query_option = st.sidebar.selectbox(
        "Select an analytics query:",
        ["None", "Top 10 Customers by Spending", "Average Order Value", "Total Revenue"]
    )

    query_mapping = {
        "Top 10 Customers by Spending": """
            SELECT customer_id, total_price 
            FROM `brave-drummer-454602-n7.ecommercedataset.dim_orders`
            ORDER BY total_price DESC
            LIMIT 10;
        """,
        "Average Order Value": """
            SELECT ROUND(SUM(total_price) / COUNT(DISTINCT(order_id)), 2) AS avg_order_value
            FROM `brave-drummer-454602-n7.ecommercedataset.dim_orders`;
        """,
        "Total Revenue": """
            SELECT ROUND(SUM(total_price), 2) AS actual_price
            FROM `brave-drummer-454602-n7.ecommercedataset.dim_orders`;
        """
    }

    if query_option != "None":
        query = query_mapping[query_option]
        df_query = fetch_bigquery_data(query)
        if df_query is not None:
            st.write(f"### {query_option} Results")
            st.dataframe(df_query)

if __name__ == "__main__":
    main()


