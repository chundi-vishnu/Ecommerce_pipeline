from google.cloud import bigquery
import os

# Set up project details
project_id = "brave-drummer-454602-n7"
dataset_id = "ecommercedataset"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\HP\Downloads\brave-drummer-454602-n7-0ec1f7058e09.json"

# Initialize BigQuery client
client = bigquery.Client(project=project_id)

def query():
    sql_query = " select customer_id,total_price From `brave-drummer-454602-n7.ecommercedataset.dim_orders` order by total_price desc limit 10"

    try:
        # Execute the query
        query_job = client.query(sql_query)
        results = query_job.result()  # Fetch the results

        # Print column names
        columns = [field.name for field in results.schema]
        print(" | ".join(columns))  # Header row
        print("-" * 50)

        # Print each row
        for row in results:
            print(" | ".join(str(row[col]) for col in columns))

    except Exception as e:
        print(f"❌ Error: {e}")  # Prints the actual error message

    

    sql_query1 = " select round(sum(total_price) / count(distinct(order_id)), 2) as avg_order_value from `brave-drummer-454602-n7.ecommercedataset.dim_orders`; "

    try:
        # Execute the query
        query_job = client.query(sql_query1)
        results = query_job.result()  # Fetch the results

        # Print column names
        columns = [field.name for field in results.schema]
        print(" | ".join(columns))  # Header row
        print("-" * 50)

        # Print each row
        for row in results:
            print(" | ".join(str(row[col]) for col in columns))

    except Exception as e:
        print(f"❌ Error: {e}")  # Prints the actual error message

    sql_query2= " select round(sum(total_price),2) as actual_price from `brave-drummer-454602-n7.ecommercedataset.dim_orders`;  "  

    try:
        # Execute the query
        query_job = client.query(sql_query2)
        results = query_job.result()  # Fetch the results

        # Print column names
        columns = [field.name for field in results.schema]
        print(" | ".join(columns))  # Header row
        print("-" * 50)

        # Print each row
        for row in results:
            print(" | ".join(str(row[col]) for col in columns))

    except Exception as e:
        print(f"❌ Error: {e}")  # Prints the actual error message

# query()


    


