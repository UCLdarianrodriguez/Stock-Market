""" This is the main program """

from acquisition import StockDataset
from store import store_data
from clean import clean_data
from exploration import eda



def main():
    """This method should be called when the program is run from the command line.
    The aim of the method is to run the complete, automated workflow you developed
    to solve the assignment.

    This function will be called by the automated test suite, so make sure that
    the function signature is not changed, and that it does not require any
    user input.

    If your workflow requires mongoDB (or any other) credentials, please commit them to
    this repository.
    Remember that if the workflow pushed new data to a mongo database without checking
    if the data is already present, the database will contain copies of the data and
    skew the results.

    After having implemented the method, please delete this docstring and replace
    it with a description of what your main method does.

    Hereafter, we provide a **volountarily suboptimal** example of how to structure
    your code. You are free to use this structure, and encouraged to improve it.

    Example:
        def main():
            # acquire the necessary data
            data = acquire()

            # store the data in MongoDB Atlas or Oracle APEX
            store(data)

            # format, project and clean the data
            proprocessed_data = preprocess(data)

            # perform exploratory data analysis
            statistics = explore(proprocessed_data)

            # show your findings
            visualise(statistics)

            # create a model and train it, visualise the results
            model = fit(proprocessed_data)
            visualise(model)
    """

    # Define general parameters for the Queries to API
    start_date = '2019-04-01'
    end_date =  '2023-04-30'
    split_day = '2023-04-01' # start day of test set
    symbol = 'MSFT'

    # Create object for acquiring the data
    stock_market = StockDataset(start_date,end_date,symbol)

    # Collection of pandas dataframes with the data
    df_stock_prices = stock_market.get_historical_price()
    df_aux_economic_data = stock_market.get_economic_axiliary_data()

    # Store the data in MongoDB Atlas
    store_data(df_stock_prices, df_aux_economic_data)

    # Read, format, clean the data, plot outliers and split data
    data_cleaned, training_set, test_set = clean_data(split_day)
    df_technical,df_non_redundant = eda(training_set)

if __name__ == "__main__":
    main()
