from google.cloud import bigquery
import os
import pandas as pd
import pandas_gbq
from pandas_gbq import to_gbq

# Set up BigQuery client
project_id="brave-drummer-454602-n7"
dataset_id="brave-drummer-454602-n7.ecommercedataset"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"c:\Users\HP\Downloads\brave-drummer-454602-n7-0ec1f7058e09.json"

client = bigquery.Client(project=project_id)

olist_products_cleaned_dataset=pd.read_csv(r"c:\Users\HP\Downloads\E-commerce pipeline cleaned data\olist_products_cleaned_dataset.csv")
olist_customers_cleaned_dataset=pd.read_csv(r"c:\Users\HP\Downloads\E-commerce pipeline cleaned data\olist_customers_cleaned_dataset.csv")
olist_orders_cleaned_dataset=pd.read_csv(r"c:\Users\HP\Downloads\E-commerce pipeline cleaned data\olist_orders_cleaned_dataset.csv")
olist_sellers_cleaned_dataset=pd.read_csv(r"c:\Users\HP\Downloads\E-commerce pipeline cleaned data\olist_sellers_cleaned_dataset.csv")
Event_data=pd.read_csv(r"c:\Users\HP\Desktop\commerce pipeline cleaned data\Event_Data.csv")
# Dictionary of dimension tables
tables = {
    'dim_products':olist_products_cleaned_dataset,
    'dim_customers':olist_customers_cleaned_dataset,
    'dim_orders':olist_orders_cleaned_dataset,
    'dim_sellers':olist_sellers_cleaned_dataset,
    'Fact_main':Event_data
}

# for table_name, df in tables.items():
#     to_gbq(df, f"{dataset_id}.{table_name}", project_id=project_id, if_exists="replace")

# # Upload all tables to BigQuery
for tab_name, df_name in tables.items():
    pandas_gbq.to_gbq(df_name, f'{dataset_id}.{tab_name}', project_id=project_id, if_exists='replace')




























































































# import mysql.connector
# import pandas as pd

# # MySQL database connection details
# DB_CONFIG = {
#     "host": "localhost",
#     "user": "root",
#     "password": "Vishnu@123",  # Ensure this is correct
#     "database": "ecommerce_project"
# }

# try:
#     # Establish connection
#     


#     if conn.is_connected():
#         print("✅ Connected to MySQL successfully!")

#         # Example: Fetch and display available tables
#         cursor.execute("SHOW TABLES;")
#         tables = cursor.fetchall()
#         print("📌 Tables in the database:", [table[0] for table in tables])

# except mysql.connector.Error as err:
#     print("❌ Error:", err)

# finally:
#     # Close cursor and connection
#     if 'cursor' in locals() and cursor:
#         cursor.close()
#     if 'conn' in locals() and conn.is_connected():
#         conn.close()
#         print("✅ MySQL connection closed.")


