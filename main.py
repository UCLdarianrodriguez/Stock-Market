""" This is the main program """

from acquisition import StockDataset
from store import store_data
from clean import clean_data
from exploration import eda
from feature_engineering import create_features
from forecasting import forecasting_models


def main():
    """

    The main function executes a series of steps to predict stock price close price

    Steps:
        1. Define general parameters for the queries to the API,
        including start and end dates, and the symbol (stock symbol).

        2. Set the starting day for the test set split.

        3. Create a StockDataset object for acquiring historical stock prices
        and economic auxiliary data.

        4. Store the acquired data in MongoDB Atlas using the store_data function.

        5. Read, format, clean the data, plot outliers, and split the data into training
        and test sets using the clean_data function.

        6. Perform exploratory data analysis (EDA) on the training set to gain
        insights into the data using the eda function.

        7. Perform feature engineering, dropping unnecessary features identified during EDA
        and create new ones using financial indicators, using create feature function

        8. Format the data for the models using the forecasting_models function, 
        fit model and plot results

    """

    # Define general parameters for the Queries to API
    start_date = "2019-04-01"
    end_date = "2023-04-30"
    symbol = "MSFT"

    # starting day for test set
    split_day = "2023-04-01"

    # Create object for acquiring the data
    stock_market = StockDataset(start_date, end_date, symbol)

    # Collection of pandas dataframes with the data
    df_stock_prices = stock_market.get_historical_price()
    df_aux_economic_data = stock_market.get_economic_axiliary_data()

    # Store the data in MongoDB Atlas
    store_data(df_stock_prices, df_aux_economic_data)

    # Read, format, clean the data, plot outliers and split data
    pd_concat, training_set, test_set = clean_data(split_day)

    # perform exploratory data analysis
    list_drop_features = eda(training_set)

    # Perform feature engineering
    dataset, train_set, test_set = create_features(
        pd_concat, list_drop_features, split_day
    )

    # Forescasting and format data for the model
    forecasting_models(dataset, train_set, test_set)


if __name__ == "__main__":
    main()
