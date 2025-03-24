import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, pandas as pd

def data_cleaning():
    olist_customers_dataset = pd.read_csv(r'C:\Users\HP\Ecommerce_pipeline\data\olist_customers_dataset.csv')
    olist_geolocation_dataset = pd.read_csv(r'C:\Users\HP\Ecommerce_pipeline\data\olist_geolocation_dataset.csv')
    olist_orders_dataset = pd.read_csv(r'C:\Users\HP\Ecommerce_pipeline\data\olist_orders_dataset.csv')
    olist_order_items_dataset = pd.read_csv(r'C:\Users\HP\Ecommerce_pipeline\data\olist_order_items_dataset.csv')
    olist_order_payments_dataset = pd.read_csv(r'C:\Users\HP\Ecommerce_pipeline\data\olist_order_payments_dataset.csv')
    olist_order_reviews_dataset = pd.read_csv(r'C:\Users\HP\Ecommerce_pipeline\data\olist_order_reviews_dataset.csv')
    olist_products_dataset = pd.read_csv(r'C:\Users\HP\Ecommerce_pipeline\data\olist_products_dataset.csv')
    olist_sellers_dataset = pd.read_csv(r'C:\Users\HP\Ecommerce_pipeline\data\olist_sellers_dataset.csv')
    product_category_name_translation = pd.read_csv(r'C:\Users\HP\Ecommerce_pipeline\data\product_category_name_translation.csv')
    print("*"*20+"olist_customer_dataset"+"*"*20)
    print(olist_customers_dataset.isna().sum())
    print(olist_geolocation_dataset.isna().sum())
    print(olist_orders_dataset.isna().sum())
    olist_orders_dataset.drop(["order_delivered_carrier_date", "order_delivered_customer_date"], axis = 1, inplace = True)
    # Deleting carrier date and customer date columns as it is not useful for analysis because of containing estimated delivery date column
    # Not handling null values in order_approved_at column as the order was cancelled before approving.
    # Checking for missing values in Olist datasets
    print(olist_orders_dataset.isna().sum())

    print(olist_order_items_dataset.isna().sum())
    print(olist_order_payments_dataset.isna().sum())
    print(olist_order_reviews_dataset.isna().sum())

    print(olist_order_reviews_dataset)

    # Dropping 'review_comment_title' due to too many null values and 'review_comment_message' as the data is inconsistent.
    olist_order_reviews_dataset.drop(["review_comment_title", "review_comment_message"], axis=1, inplace=True)

    print(olist_order_reviews_dataset.isna().sum())

    print(olist_products_dataset.isna().sum())

    print(olist_products_dataset)

    # Finding rows where 'product_category_name' is missing
    print(olist_products_dataset[olist_products_dataset["product_category_name"].isna()])

    # Deleting null rows in product dataset
    olist_products_dataset.drop(list(olist_products_dataset[(olist_products_dataset["product_name_lenght"].isna())].index),
                                axis=0, 
                                inplace=True)

    print(olist_products_dataset[olist_products_dataset["product_weight_g"].isna()])

    # Dropping rows with missing 'product_weight_g'
    olist_products_dataset.drop(olist_products_dataset[olist_products_dataset["product_weight_g"].isna()].index.tolist(),
                                axis=0,
                                inplace=True)

    # Dropping specific product categories
    olist_products_dataset.drop(olist_products_dataset[olist_products_dataset["product_category_name"].isin(
        ["pc_gamer", "portateis_cozinha_e_preparadores_de_alimentos"])].index.tolist(),
                                axis=0,
                                inplace=True)

    # Creating a dictionary for product category name translations
    names_trans = {}

    for i, j in zip(product_category_name_translation["product_category_name"], product_category_name_translation["product_category_name_english"]):
        names_trans[i] = j
    # Mapping translated product category names
    names = []

    for i in olist_products_dataset["product_category_name"]:
        names.append(names_trans.get(i))

    olist_products_dataset.insert(2, "product_category", names)

    # Dropping the old column after translation
    olist_products_dataset.drop("product_category_name", axis=1, inplace=True)

    print(olist_products_dataset.isna().sum())

    # Renaming columns for consistency
    olist_products_dataset = olist_products_dataset.rename(columns={
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length",
        "product_photos_qty": "product_photos_quantity",
        "product_weight_g": "product_weight_gm"
    })

    print(olist_products_dataset.isna().sum())

    print(olist_sellers_dataset.isna().sum())

    print(product_category_name_translation.isna().sum())

    # Checking and removing duplicates
    print(olist_customers_dataset.duplicated().sum())
    print(olist_geolocation_dataset.duplicated().sum())

    olist_geolocation_dataset.drop_duplicates(inplace=True)
    print(olist_geolocation_dataset.duplicated().sum())

    # Aggregating order items data
    olist_order_items_agg = olist_order_items_dataset.groupby("order_id").agg(
        total_items=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight_value=("freight_value", "sum")
    ).reset_index()

    # Aggregating order payments data
    olist_order_payments_agg = olist_order_payments_dataset.groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        payment_types_used=("payment_type", lambda x: x.mode().iloc[0]),
        max_installments=("payment_installments", "max")
    ).reset_index()

    # Aggregating order reviews data
    olist_order_reviews_agg = olist_order_reviews_dataset.groupby("order_id").agg(
        avg_review_score=("review_score", "mean"),
    ).reset_index()

    '''
    Pushing this useful aggregated data to the olist_orders dataset because 
    order_items, order_payments, and order_reviews do not have a unique column.
    '''
    '''
    Pushing useful aggregated data from order_items, order_payments, and order_reviews 
    to olist_orders_dataset as they lack a unique column.
    '''

    # Merging aggregated datasets into olist_orders_dataset
    olist_orders_dataset = pd.merge(olist_orders_dataset, olist_order_items_agg, on="order_id", how="left")
    olist_orders_dataset = pd.merge(olist_orders_dataset, olist_order_payments_agg, on="order_id", how="left")
    olist_orders_dataset = pd.merge(olist_orders_dataset, olist_order_reviews_agg, on="order_id", how="left")

    # Checking for missing values
    print(olist_orders_dataset.isna().sum())

    # Handling missing values in 'total_items'
    print(olist_orders_dataset[olist_orders_dataset["total_items"].isna()])

    # Dropping rows where 'total_items' is missing
    olist_orders_dataset.drop(olist_orders_dataset[olist_orders_dataset["total_items"].isna()].index.tolist(),
                            axis=0, inplace=True)

    print(olist_orders_dataset.isna().sum())

    # Checking and removing missing values in 'total_payment_value'
    print(olist_orders_dataset[olist_orders_dataset["total_payment_value"].isna()])
    print(olist_orders_dataset.drop(30710, axis=0, inplace=True))

    # Handling missing values in 'avg_review_score'
    print(olist_orders_dataset[olist_orders_dataset["avg_review_score"].isna()])

    # Calculating Interquartile Range (IQR) to find outliers
    Q1 = olist_orders_dataset["avg_review_score"].quantile(0.25)
    Q3 = olist_orders_dataset["avg_review_score"].quantile(0.75)
    IQR = Q3 - Q1
    LL = Q1 - (1.5 * IQR)  # Lower Limit
    UL = Q3 + (1.5 * IQR)  # Upper Limit

    # Finding outliers in 'avg_review_score'
    olist_orders_dataset[(olist_orders_dataset["avg_review_score"] < LL) | 
                        (olist_orders_dataset["avg_review_score"] > UL)]

    # Filling missing 'avg_review_score' values with median
    olist_orders_dataset["avg_review_score"].fillna(olist_orders_dataset["avg_review_score"].median(), inplace=True)

    olist_orders_dataset.isna().sum()

    # Renaming 'payment_types_used' to 'payment_type'
    olist_orders_dataset = olist_orders_dataset.rename(columns={"payment_types_used": "payment_type"})

    # Checking for duplicates
    print(olist_orders_dataset.duplicated().sum())
    print(olist_order_reviews_dataset.duplicated().sum())
    print(olist_orders_dataset.duplicated().sum())
    print(olist_products_dataset.duplicated().sum())
    print(olist_sellers_dataset.duplicated().sum())
    print(product_category_name_translation.duplicated().sum())

    # Checking data types and structure of various datasets
    print(olist_customers_dataset.dtypes)
    print(olist_customers_dataset.head())

    print(olist_geolocation_dataset.dtypes)
    print(olist_geolocation_dataset.head())

    print(olist_order_items_dataset.dtypes)
    print(olist_order_items_dataset.head())
    # Converting datetime columns to proper datetime format
    olist_order_items_dataset["shipping_limit_date"] = pd.to_datetime(olist_order_items_dataset["shipping_limit_date"])

    print(olist_order_payments_dataset.dtypes)
    print(olist_order_payments_dataset.head())

    print(olist_order_reviews_dataset.dtypes)
    print(olist_order_reviews_dataset.head())

    olist_order_reviews_dataset["review_creation_date"] = pd.to_datetime(olist_order_reviews_dataset["review_creation_date"])
    olist_order_reviews_dataset["review_answer_timestamp"] = pd.to_datetime(olist_order_reviews_dataset["review_answer_timestamp"])

    print(olist_orders_dataset.dtypes)
    print(olist_orders_dataset.head())

    olist_orders_dataset["order_purchase_timestamp"] = pd.to_datetime(olist_orders_dataset["order_purchase_timestamp"])
    olist_orders_dataset["order_approved_at"] = pd.to_datetime(olist_orders_dataset["order_approved_at"])
    olist_orders_dataset["order_estimated_delivery_date"] = pd.to_datetime(olist_orders_dataset["order_estimated_delivery_date"])

    # Converting numeric columns to optimized data types
    olist_orders_dataset["total_items"] = olist_orders_dataset["total_items"].astype("int8")
    olist_orders_dataset["max_installments"] = olist_orders_dataset["max_installments"].astype("int8")
    olist_orders_dataset["avg_review_score"] = olist_orders_dataset["avg_review_score"].astype("int8")

    print(olist_products_dataset.dtypes)
    print(olist_products_dataset.head())

    # Optimizing product-related numerical columns
    olist_products_dataset["product_name_length"] = olist_products_dataset["product_name_length"].astype("int16")
    olist_products_dataset["product_description_length"] = olist_products_dataset["product_description_length"].astype("int16")
    olist_products_dataset["product_photos_quantity"] = olist_products_dataset["product_photos_quantity"].astype("int16")
    olist_products_dataset["product_weight_gm"] = olist_products_dataset["product_weight_gm"].astype("int16")
    olist_products_dataset["product_length_cm"] = olist_products_dataset["product_length_cm"].astype("int16")
    olist_products_dataset["product_height_cm"] = olist_products_dataset["product_height_cm"].astype("int16")
    olist_products_dataset["product_width_cm"] = olist_products_dataset["product_width_cm"].astype("int16")
    print(olist_sellers_dataset.dtypes)
    print(olist_sellers_dataset.head())

    print(product_category_name_translation.dtypes)

    print("*"*20+"Exploratory Data Analysis On Olist Dataset"+"*"*20)
    print("olist_customers_dataset")
    print(olist_customers_dataset)
    discrete_cols = ["customer_state"]

    for i in discrete_cols:
        print("*"*50 + " "+ f"{i}" +" "+"*"*50)
        plt.figure(figsize = (6,6))
        olist_customers_dataset[i].value_counts().plot.barh()
        plt.xlabel("Count")
        plt.ylabel(i)
        plt.show()
        print()
    print("")
    print(olist_geolocation_dataset)
    discrete_cols = ["geolocation_state"]
    print("discrete")
    for i in discrete_cols:
        print("*"*50 + " "+ f"{i}" +" "+"*"*50)
        plt.figure(figsize = (6,6))
        olist_geolocation_dataset[i].value_counts().plot.barh()
        plt.xlabel("Count")
        plt.ylabel(i)
        plt.show()
        print()
    
    print(olist_order_items_dataset)
    print("continue")
    continuous_cols = ["price", "freight_value"]

    for i in continuous_cols:
        print("*"*50 + " "+ f"{i}" +" "+"*"*50)
        plt.figure(figsize = (6,6))
        olist_order_items_dataset[i].plot.kde()
        plt.xlabel(i)
        plt.title(f"Checking the distribution of {i} Column")
        plt.show()
        print()
    continuous_cols = ["price", "freight_value"]

    for i in continuous_cols:
        print("*"*50 + " "+ f"{i}" +" "+"*"*50)
        plt.figure(figsize = (6,6))
        olist_order_items_dataset[i].plot.box()
        plt.title(f"Checking Outliers in {i} Column")
        plt.show()
        print()
    print(olist_order_payments_dataset)

    discrete_cols = ["payment_type", "payment_installments", "payment_sequential"]

    for i in discrete_cols:
        print("*"*50 + " "+ f"{i}" +" "+"*"*50)
        plt.figure(figsize = (6,6))
        olist_order_payments_dataset[i].value_counts().plot.barh()
        plt.xlabel("Count")
        plt.ylabel(i)
        plt.show()
        print()
    
    continuous_cols = ["payment_value"]

    for i in continuous_cols:
        print("*"*50 + " "+ f"{i}" +" "+"*"*50)
        plt.figure(figsize = (6,6))
        olist_order_payments_dataset[i].plot.kde()
        plt.xlabel(i)
        plt.title(f"Checking the distribution of {i} Column")
        plt.show()
        print()
    
    continuous_cols = ["payment_value"]

    for i in continuous_cols:
        print("*"*50 + " "+ f"{i}" +" "+"*"*50)
        plt.figure(figsize = (6,6))
        olist_order_payments_dataset[i].plot.box()
        plt.show()
        print()
    print(olist_order_reviews_dataset)
    discrete_cols = ["review_score"]

    for i in discrete_cols:
        print("*"*50 + " "+ f"{i}" +" "+"*"*50)
        plt.figure(figsize = (6,6))
        olist_order_reviews_dataset[i].value_counts().plot.barh()
        plt.xlabel("Count")
        plt.ylabel(i)
        plt.show()
        print()
    for i in discrete_cols:
        print("*"*50 + " "+ f"{i}" +" "+"*"*50)
        plt.figure(figsize = (6,6))
        olist_order_reviews_dataset[i].value_counts().plot.pie(autopct = "%.2f")
        plt.show()
    print(olist_orders_dataset)

    discrete_cols = ["order_status", "total_items", "payment_type", "max_installments", "avg_review_score"]

    for i in discrete_cols:
        print("*"*50 + " "+ f"{i}" +" "+"*"*50)
        plt.figure(figsize = (6,6))
        olist_orders_dataset[i].value_counts().plot.barh()
        plt.xlabel("Count")
        plt.ylabel(i)
        plt.show()
        print()
    print(olist_orders_dataset)
    continuous_cols = ["total_price", "total_freight_value", "total_payment_value"]

    for i in continuous_cols:
        print("*"*50 + " "+ f"{i}" +" "+"*"*50)
        plt.figure(figsize = (6,6))
        olist_orders_dataset[i].plot.kde()
        plt.xlabel(i)
        plt.title(f"Checking the distribution of {i} Column")
        plt.show()
        print()
    for i in continuous_cols:
        print("*"*50 + " "+ f"{i}" +" "+"*"*50)
        plt.figure(figsize = (6,6))
        olist_orders_dataset[i].plot.box()
        plt.show()
        print()
    print(olist_products_dataset)

    continuous_cols = ["product_name_length", "product_description_length", "product_weight_gm", "product_length_cm","product_height_cm","product_width_cm"]

    for i in continuous_cols:
        print("*"*50 + " "+ f"{i}" +" "+"*"*50)
        plt.figure(figsize = (6,6))
        olist_products_dataset[i].plot.kde()
        plt.xlabel(i)
        plt.title(f"Checking the distribution of {i} Column")
        plt.show()
        print()
    continuous_cols = ["product_name_length", "product_description_length", "product_weight_gm", "product_length_cm","product_height_cm","product_width_cm"]

    for i in continuous_cols:
        print("*"*50 + " "+ f"{i}" +" "+"*"*50)
        plt.figure(figsize = (6,6))
        olist_products_dataset[i].plot.box()
        plt.show()
        print()
    discrete_cols = ["product_photos_quantity"]

    for i in discrete_cols:
        print("*"*50 + " "+ f"{i}" +" "+"*"*50)
        plt.figure(figsize = (6,6))
        olist_products_dataset[i].value_counts().plot.barh()
        plt.xlabel("Count")
        plt.ylabel(i)
        plt.show()
        print()
    plt.figure(figsize = (6,6))
    olist_sellers_dataset["seller_state"].value_counts().plot.barh()
    plt.show()
    print("saving the cleaned datasets")
    olist_customers_dataset.to_csv(r"c:\Users\HP\Downloads\E-commerce pipeline cleaned data\olist_customers_cleaned_dataset.csv", index = None)
    olist_geolocation_dataset.to_csv(r"c:\Users\HP\Downloads\E-commerce pipeline cleaned data\olist_geolocation_cleaned_dataset.csv", index = None)
    olist_order_items_dataset.to_csv(r"c:\Users\HP\Downloads\E-commerce pipeline cleaned data\olist_order_items_cleaned_dataset.csv", index = None)
    olist_order_payments_dataset.to_csv(r"c:\Users\HP\Downloads\E-commerce pipeline cleaned data\olist_order_payments_cleaned_dataset.csv", index = None)
    olist_order_reviews_dataset.to_csv(r"c:\Users\HP\Downloads\E-commerce pipeline cleaned data\olist_order_reviews_cleaned_dataset.csv", index = None)
    olist_orders_dataset.to_csv(r"c:\Users\HP\Downloads\E-commerce pipeline cleaned data\olist_orders_cleaned_dataset.csv", index = None)
    olist_products_dataset.to_csv(r"c:\Users\HP\Downloads\E-commerce pipeline cleaned data\olist_products_cleaned_dataset.csv", index = None)
    olist_sellers_dataset.to_csv(r"c:\Users\HP\Downloads\E-commerce pipeline cleaned data\olist_sellers_cleaned_dataset.csv", index = None)
    
    cleaned_files = os.listdir(r"c:\Users\HP\Downloads\E-commerce pipeline cleaned data")
    l = []
    for i in cleaned_files:
        l.append(pd.read_csv(r"c:\Users\HP\Downloads\E-commerce pipeline cleaned data\{}".format(i)))
    print(cleaned_files)

    olist_customers_cleaned_dataset = l[0]

    olist_geolocation_cleaned_dataset = l[1]

    olist_orders_cleaned_dataset = l[2]

    olist_order_items_cleaned_dataset = l[3]

    olist_order_payments_cleaned_dataset = l[4]

    olist_order_reviews_cleaned_dataset = l[5]

    olist_products_cleaned_dataset = l[6]

    olist_sellers_cleaned_dataset = l[7]





    
    print(olist_customers_dataset)
    print(olist_geolocation_dataset)
    print(olist_order_payments_dataset)
    print(olist_order_reviews_dataset)
    print(olist_orders_dataset)
    print(olist_products_dataset)
    print(olist_sellers_dataset)
    print(olist_orders_dataset)


def data_cleaningEvent():
    import json
    import random
    from faker import Faker

    # Load Olist datasets
    olist_products_df = pd.read_csv(r"c:\Users\HP\Downloads\E-commerce pipeline cleaned data\olist_products_cleaned_dataset.csv")
    olist_orders_df = pd.read_csv(r"c:\Users\HP\Downloads\E-commerce pipeline cleaned data\olist_orders_cleaned_dataset.csv")
    olist_order_items_df = pd.read_csv(r"c:\Users\HP\Downloads\E-commerce pipeline cleaned data\olist_order_items_cleaned_dataset.csv")
    olist_customers_df = pd.read_csv(r"c:\Users\HP\Downloads\E-commerce pipeline cleaned data\olist_customers_cleaned_dataset.csv")

    # Extract lists of real data
    product_ids = olist_products_df["product_id"].tolist()
    order_ids = olist_orders_df["order_id"].tolist()
    customer_ids = olist_customers_df["customer_id"].tolist()
    seller_ids = olist_order_items_df["seller_id"].tolist()

    fake = Faker()
    Faker.seed(42)

    # Define event types
    event_types = ["purchase"]
    payment_types = ["credit_card", "debit_card", "voucher", "boleto"]
    payment_weights = [0.6, 0.2, 0.1, 0.1]  # Adjusted distribution

    def generate_event():
        event_type = random.choice(event_types)

        # Select a real product, order, customer, and seller
        product_id = random.choice(product_ids)
        order_id = random.choice(order_ids) if event_type in ["add_to_cart", "remove_from_cart", "purchase"] else None
        customer_id = random.choice(customer_ids)
        seller_id = random.choice(seller_ids)

        event = {
            "event_id": fake.uuid4(),
            "timestamp": fake.iso8601(),
            "customer_id": customer_id, 
            "session_id": fake.uuid4(),
            "order_id": order_id,
            "product_id": product_id,
            "seller_id": seller_id,
            "product_category": fake.word(),
            "product_price": round(random.uniform(10, 1000), 2),
            "quantity": (
                random.randint(1, 5) if event_type in ["add_to_cart", "remove_from_cart", "purchase"]
                else (1 if event_type == "product_view" else 0)
            ),
            "payment_type": (
                random.choices(payment_types, weights=payment_weights, k=1)[0]
                if event_type == "purchase"
                else None
            ),
            "customer_state": fake.state_abbr() if event_type == "purchase" else "Unknown",
            "customer_city": fake.city() if event_type == "purchase" else "Unknown"
        }
        return event
    data = pd.read_json("synthetic_ecommerce_events.json")
    data.info()
    print("Handling missing values on synthetic Event Generator")
    print(data.isna().sum())
    print(data.duplicated().sum())
    data["quantity"] = data["quantity"].astype("int8")
    print(data.dtypes)
    print(data.head())
    print("EDA on Synthetic EVent Generator")
    print(data)
    discrete_cols = ["payment_type", "quantity"]

    for i in discrete_cols:
        print("*"*50 + " "+ f"{i}" +" "+"*"*50)
        plt.figure(figsize = (6,6))
        data[i].value_counts().plot.barh()
        plt.xlabel("Count")
        plt.ylabel(i)
        plt.show()
        print()
    
    for i in discrete_cols:
        print("*"*50 + " "+ f"{i}" +" "+"*"*50)
        plt.figure(figsize = (6,6))
        data[i].value_counts().plot.pie(autopct = "%.2f")
        plt.ylabel("")
        plt.show()
        print()
    data["product_price"].plot.kde(color = "red")
    plt.xlabel("Product Price")
    plt.title("Checking the distribution of Product Price Column")
    plt.show()

    sns.boxplot(data = data, x = "product_price")
    plt.title("Checking the Outliers in Product Price Column")
    plt.show()

    c = 1
    for i in discrete_cols:
        plt.figure(figsize = (6,10))
        plt.subplot(4,1,c)
        sns.boxplot(data = data, x = i, y = "product_price")
        plt.xlabel(i)
        plt.ylabel("Product Price")
        plt.xticks(fontsize = 8)
        plt.title(f"Relation between Product Price column and {i} column")
        plt.show()
        print("*"*100)

    sns.countplot(data = data, x = "payment_type", hue = "quantity")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()



    if __name__ == "__main__":
        num_events = 30000  
        events = [generate_event() for _ in range(num_events)]
        
        with open("synthetic_ecommerce_events.json", "w") as f:
            json.dump(events, f, indent=4)

        print("Synthetic e-commerce events with real product, order, customer, and seller IDs generated successfully!")


# data_cleaning()
# data_cleaningEvent()

    