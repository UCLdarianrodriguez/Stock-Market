""" This program contains the function to store the data
    retrieved from the API and connect to mongoDB
"""

import pandas as pd
from pymongo import MongoClient
from pymongo.collection import Collection


def connect() -> MongoClient:
    """Function to connect with MongoDB with defined uri

    Returns:
    (MongoClient): Mongodb client
    """

    # Connection string for MongoDB
    password = "daps1234"
    uri = (
        "mongodb+srv://root:"
        + password
        + "@cluster0.g8gdrza.mongodb.net/?retryWrites=true&w=majority"
    )

    # Create a new client and connect to the server
    client = MongoClient(uri)

    # Send a ping to confirm a successful connection
    try:
        client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")

    except Exception as client_error:
        print(client_error)

    return client


def connect_collections() -> Collection:
    """
    Connect to the expected collections Stock and Auxiliary
    from db_MSFT in mongodb

    Returns:
        stock_prices: collection named Stock (stock prices)
        stock_aux: collection named Auxiliary
    """

    # Create client instance
    client = connect()

    # Connecting to database
    database = client["db_MSFT"]

    # verify existing collections
    list_collections = database.list_collection_names()
    collections_name = ["Stock", "Auxiliary"]
    all_present = all(item in list_collections for item in collections_name)

    assert all_present, "Collections does not exist in database"

    # Getting collection instance
    stock_prices = database["Stock"]
    stock_aux = database["Auxiliary"]

    return stock_prices, stock_aux


def store_data(df_stock_prices: pd.DataFrame, df_aux_economic_data: pd.DataFrame):
    """
    Store data in Mongodb

    Args:
        df_stock_prices (str): endpoint url
        df_aux_economic_data (dict): dictionary with the query parameters

    Returns:
        None
    """

    # Define options for the time series collection
    time_series_options = {
        "timeseries": {
            "timeField": "date",
            "granularity": "hours",  # Adjust granularity
        }
    }

    # Create client instance
    client = connect()

    # Create database if does not exist
    database = client["db_MSFT"]

    # Create collections or read
    #stock_prices = database["Stock"]
    #stock_aux = database["Auxiliary"]

    # Handling columns with strings to convert to numbers, excluding date
    for df_column in df_aux_economic_data.columns[1:]:
        # Convert columns to numeric values, handling NaN values
        df_aux_economic_data[df_column] = pd.to_numeric(
            df_aux_economic_data[df_column], errors="coerce"
        )

    # Convert DataFrame to a list of dictionaries (each dictionary represents a document)
    stock_dict = df_stock_prices.to_dict(orient="records")
    aux_dict = df_aux_economic_data.to_dict(orient="records")

    # verify existing collections
    list_collections = database.list_collection_names()
    collections_name = ["Stock", "Auxiliary"]
    #all_present = any(item in list_collections for item in collections_name)

    # Delete each collection
    for name in collections_name:
        collection = database[name]
        collection.drop()

    #assert not all_present, "Collections already exist in the database"

    # Create timeseries collections
    stock_prices = database.create_collection("Stock", **time_series_options)
    stock_aux = database.create_collection("Auxiliary", **time_series_options)

    # Insert data into their collections
    stock_prices.insert_many(stock_dict)
    stock_aux.insert_many(aux_dict)

    # Close connection with the client
    client.close()
