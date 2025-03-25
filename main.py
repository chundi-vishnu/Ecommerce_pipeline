import streamlit as st
import pandas as pd
import os
from google.cloud import bigquery
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.data_cleaning import data_cleaning


# Set up project details
project_id = "brave-drummer-454602-n7"
dataset_id = "ecommercedataset"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\HP\Downloads\brave-drummer-454602-n7-0ec1f7058e09.json"

# Initialize BigQuery client
client = bigquery.Client(project=project_id)

# Configure Streamlit Page
st.set_page_config(layout="wide", page_title="E-commerce Dashboard", page_icon="📊")


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


def run_query(query):
    try:
        query_job = client.query(query)
        results = query_job.result()
        return results.to_dataframe()
    except Exception as e:
        st.error(f"BigQuery Error: {e}")
        return None

# Sidebar for Navigation
st.sidebar.header("📊 E-commerce Dashboard")
page = st.sidebar.radio("Select a Page:", ["Data Cleaning","EDA","Analysis", "KPI Metrics", "Aggregate Tables"])

# ----------------- Data Cleaning Page -----------------
if page == "Data Cleaning":
    st.header("🛠️ Data Cleaning Process")
    if st.button("Run Data Cleaning"):
        with st.spinner("Cleaning Data..."):
            data_cleaning()
        st.success("✅ Data Cleaning Completed!")

# ----------------- KPI Metrics Page -----------------
elif page == "KPI Metrics":
    st.header("📈 Key Performance Indicators")
    
    queries = {
        "Average Revenue": """SELECT ROUND(SUM(total_price) / COUNT(DISTINCT(order_id)), 2) AS avg_order_value 
                               FROM `brave-drummer-454602-n7.ecommercedataset.dim_orders`;""",
        "Total Revenue": """SELECT ROUND(SUM(total_price),2) AS actual_price 
                            FROM `brave-drummer-454602-n7.ecommercedataset.dim_orders`;""",
        "Top Revenue Products": """SELECT p.product_id, p.product_category, SUM(o.total_price) AS total_revenue 
                FROM `brave-drummer-454602-n7.ecommercedataset.dim_products` p 
                INNER JOIN `brave-drummer-454602-n7.ecommercedataset.Fact_main` f ON p.product_id = f.product_id 
                INNER JOIN `brave-drummer-454602-n7.ecommercedataset.dim_orders` o ON f.order_id = o.order_id 
                GROUP BY p.product_id, p.product_category 
                ORDER BY total_revenue DESC 
                LIMIT 5;""",
        "Top 10 Customers by Spending": """
             SELECT customer_id, total_price 
             FROM `brave-drummer-454602-n7.ecommercedataset.dim_orders`
             ORDER BY total_price DESC
             LIMIT 10;
         """
    }
    
    for metric, query in queries.items():
        st.subheader(f"📊 {metric}")
        df = run_query(query)
        if df is not None:
            st.dataframe(df)

# ----------------- Aggregate Tables Page -----------------
elif page == "Aggregate Tables":
    st.header("📋 Aggregate Reports")
    
    queries = {
        "Daily Sales Summary": """SELECT DATE(order_purchase_timestamp) AS order_date, 
                                  COUNT(DISTINCT(order_id)) AS total_orders, 
                                  ROUND(SUM(total_price), 2) AS total_revenue 
                                  FROM `brave-drummer-454602-n7.ecommercedataset.dim_orders` 
                                  GROUP BY order_date ORDER BY order_date;""",
        "Top Selling Products": """SELECT p.product_id, p.product_category, SUM(f.product_price) AS total_revenue, 
                                 COUNT(DISTINCT f.order_id) AS total_orders, SUM(f.quantity) AS total_quantity 
                                 FROM `brave-drummer-454602-n7.ecommercedataset.dim_products` p 
                                 JOIN `brave-drummer-454602-n7.ecommercedataset.Fact_main` f ON p.product_id = f.product_id 
                                 GROUP BY p.product_id, p.product_category ORDER BY total_revenue DESC;""",
        "Customer Purchase Behavior": """SELECT c.customer_id, COUNT(DISTINCT o.order_id) AS total_orders, 
                                       SUM(f.product_price) AS total_spent, AVG(f.product_price) AS avg_order_value, 
                                       MAX(o.order_purchase_timestamp) AS last_purchase_date 
                                       FROM `brave-drummer-454602-n7.ecommercedataset.dim_customers` c 
                                       JOIN `brave-drummer-454602-n7.ecommercedataset.Fact_main` f ON c.customer_id = f.customer_id 
                                       JOIN `brave-drummer-454602-n7.ecommercedataset.dim_orders` o ON f.order_id = o.order_id 
                                       GROUP BY c.customer_id ORDER BY total_spent DESC;""",
        "Seller Performance": """SELECT s.seller_id, COUNT(DISTINCT o.order_id) AS total_orders, 
                               SUM(o.total_price) AS total_revenue, 
                               ROUND(AVG(DATE_DIFF(DATE(o.order_estimated_delivery_date), DATE(o.order_approved_at), DAY)), 2) AS avg_delivery_time 
                               FROM `brave-drummer-454602-n7.ecommercedataset.dim_sellers` s 
                               JOIN `brave-drummer-454602-n7.ecommercedataset.Fact_main` f ON s.seller_id = f.seller_id 
                               JOIN `brave-drummer-454602-n7.ecommercedataset.dim_orders` o ON f.order_id = o.order_id 
                               WHERE o.order_estimated_delivery_date IS NOT NULL 
                               GROUP BY s.seller_id ORDER BY total_revenue DESC;"""
    }
    
    for table, query in queries.items():
        st.subheader(f"📊 {table}")
        df = run_query(query)
        if df is not None:
            st.dataframe(df)
elif page=="EDA":
    st.header("EDA")

    #Sidebar for dataset selection
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
elif page=="Analysis":
    st.header("Analysis")
        # Additional Visualization Options
    st.sidebar.title("Additional Analysis")
    dataset_paths = {
        "Customers": "c:/Users/HP/Downloads/E-commerce pipeline cleaned data/olist_customers_cleaned_dataset.csv",
        "Orders": "c:/Users/HP/Downloads/E-commerce pipeline cleaned data/olist_orders_cleaned_dataset.csv",
        "Products": "c:/Users/HP/Downloads/E-commerce pipeline cleaned data/olist_products_cleaned_dataset.csv",
        "Payments": "c:/Users/HP/Downloads/E-commerce pipeline cleaned data/olist_order_payments_cleaned_dataset.csv",
        "Reviews": "c:/Users/HP/Downloads/E-commerce pipeline cleaned data/olist_order_reviews_cleaned_dataset.csv",
        "Sellers": "c:/Users/HP/Downloads/E-commerce pipeline cleaned data/olist_sellers_cleaned_dataset.csv"
    }
    analysis_option = st.sidebar.selectbox(
        "Select an analysis type:",
        ["None", "Order Status Distribution", "Revenue by Payment Type", "Customer Distribution","Daily Orders & Revenue"]
    )

    if analysis_option == "Order Status Distribution":
        st.title("Order Status Distribution")
        df = load_dataset(dataset_paths["Orders"])
        if df is not None:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x=df["order_status"], palette="coolwarm", ax=ax)
            plt.xticks(rotation=45)
            plt.title("Order Status Distribution")
            plt.xlabel("Order Status")
            plt.ylabel("Count")
            st.pyplot(fig)

    elif analysis_option == "Revenue by Payment Type":
        st.title("Revenue by Payment Type")
        df = load_dataset(dataset_paths["Payments"])
        if df is not None:
            revenue_by_payment = df.groupby("payment_type")["payment_value"].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=revenue_by_payment.index, y=revenue_by_payment.values, palette="Blues_r", ax=ax)
            plt.xticks(rotation=45)
            plt.title("Total Revenue by Payment Type")
            plt.xlabel("Payment Type")
            plt.ylabel("Total Revenue")
            st.pyplot(fig)

    elif analysis_option == "Customer Distribution":
        st.title("Customer Distribution by State")
        df = load_dataset(dataset_paths["Customers"])
        if df is not None:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(y=df["customer_state"], palette="magma", 
                          order=df["customer_state"].value_counts().index, ax=ax)
            plt.title("Customer Distribution by State")
            plt.xlabel("Count")
            plt.ylabel("State")
            st.pyplot(fig)
    elif analysis_option=="Daily Orders & Revenue":
        query = """
             SELECT DATE(order_purchase_timestamp) AS order_date, 
                COUNT(DISTINCT order_id) AS total_orders, 
                ROUND(SUM(total_price),2) AS total_revenue
             FROM `brave-drummer-454602-n7.ecommercedataset.dim_orders`
             GROUP BY order_date
             ORDER BY order_date;
        """
        df_daily = run_query(query)
        if df_daily is not None:
            st.write("### Daily Orders & Revenue")
            st.dataframe(df_daily)
            
           















