"""
Class to acquire the time series historical data from differents apis
it acquires the stock prices and economic indicator as auxiliary data and
convert the result into pandas dataframe

"""


# Importing Modules
import requests
import pandas as pd


class StockDataset:

    """
    This class the data acquisition steps for the time series.

    Attributes:
    - start_date (str): the start date of the data.
    - end_date (str): the end date of the data.
    - company (str): the name of the company.

    """

    def __init__(self, start_date, end_date, company):
        self.start_date = start_date
        self.end_date = end_date
        self.company = company

        # Define api_key for alpha Vantage
        self.alpha_api_key = "RXTP6VPBN0Y6MEUT"

        # Define api_key for Financial Modeling Prep (fmp)
        self.fmp_api_key = "df2413659ea0b96031b0303329cc9be6"

        # Define api_key fred Economic Data
        self.fred_api_key = "5530c45b1ed2b14d85d841edc4fd67ee"

    def acquire_data_api(self, baseurl: str, params: dict) -> dict:
        """
        Acquires data from api given a baseurl and query parameters

        Args:
            baseurl (str): endpoint url
            params (dict): dictionary with the query parameters

        Returns:
            (dict): dictionary of data returned from the api
        """

        response = requests.get(baseurl, params, timeout=120)

        # Make sure the query was successful
        assert (
            response.status_code == 200
        ), f"Error retrieving information: {response.status_code}"

        retrieved_info = response.json()

        return retrieved_info

    def get_economic_axiliary_data(self):
        """
        Acquires Economic data from alpha vantage and fred

        Args:

        Returns:
            (pandas): pandas dataframe with the historical data
        """

        # Query parameters
        params = {
            "function": "TREASURY_YIELD",
            "apikey": self.alpha_api_key,
            "interval": "daily",
        }

        data = self.economic_data(params)

        pd_concat = data.copy()

        # Define Economic indicator list to retrieve from fred
        indicators = ["VIXCLS", "T5YIE", "INFECTDISEMVTRACKD", "DFF"]

        for indicator in indicators:
            params = {
                "observation_start": self.start_date,
                "observation_end": self.end_date,
                "api_key": self.fred_api_key,
                "series_id": indicator,
                "file_type": "json",
                "frequency": "d",
            }

            # Acquire data from api
            url = "https://api.stlouisfed.org/fred//series/observations?"

            fred_data = self.acquire_data_api(url, params)

            df_fred = pd.DataFrame(fred_data["observations"])

            # Drop useless columns
            df_fred.drop(columns=["realtime_start", "realtime_end"], inplace=True)

            # Convert index to datatime
            df_fred["date"] = pd.to_datetime(df_fred["date"])
            
            # Rename columns
            df_fred.rename(columns={"value": indicator}, inplace=True)

            # Merge DataFrames based on the 'date' column
            pd_concat = pd.merge(pd_concat, df_fred, on='date', how='inner')

        return pd_concat

    def get_historical_price(self):
        """
        Acquires MSFT historical data from fmp api

        Args:

        Returns:
            (pandas): pandas dataframe with the historical data
        """

        params = {
            "from": self.start_date,
            "to": self.end_date,
            "apikey": self.fmp_api_key,
        }

        # Historical stock price data for fmp api
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{self.company}"

        # perform the request to api
        data = self.acquire_data_api(url, params)

        # get only the historical data
        stock_data = pd.DataFrame(data["historical"])

        # Set the DataFrame's date column to be a Datetime
        stock_data["date"] = pd.to_datetime(stock_data["date"], format="%Y-%m-%d")

        # drop label column, does not add information
        stock_data.drop(["label"], axis=1, inplace=True)

        # Sort DataFrame by 'date' column in ascending order
        stock_data.sort_values(by="date", ascending=True, inplace=True)

        # Reset index after sorting and drop the all indexes
        stock_data.reset_index(drop=True, inplace=True)

        return stock_data

    def technical_data(self, params):
        """
        Acquires technical indicators from alpha Vantage api

        Args:

        Returns:
            (pandas): pandas dataframe with dates filtered
        """
        # Technical indicators url
        url = "https://www.alphavantage.co/query"

        # perform the request to api
        data = self.acquire_data_api(url, params)

        # get the tecnical indicators
        stock_data = pd.DataFrame(data[f"Technical Analysis: {params['function']}"]).T

        # Convert index to datatime
        stock_data.index = pd.to_datetime(stock_data.index)

        # Sort dates before slicing
        stock_data.sort_index(inplace=True)

        # Filter desired dates
        filtered_data = stock_data.loc[self.start_date : self.end_date, :]

        return filtered_data

    def economic_data(self, params):
        """
        Acquires economic indicators from alpha Vantage api

        Args:

        Returns:
            (pandas): pandas dataframe with dates filtered
        """

        # Economic indicators url
        url = "https://www.alphavantage.co/query"

        # perform the request to api
        data = self.acquire_data_api(url, params)

        # get Economic indicators
        stock_data = pd.DataFrame(data["data"])

        # Convert index to datatime
        stock_data["date"] = pd.to_datetime(stock_data["date"])

        # Rename columns
        stock_data.rename(columns={"value": params["function"]}, inplace=True)

        # Sort DataFrame by 'date' column in ascending order
        stock_data.sort_values(by="date", ascending=True, inplace=True)

        # Filter desired dates
        filtered_data = stock_data[
            (stock_data["date"] >= self.start_date)
            & (stock_data["date"] <= self.end_date)
        ]

        return filtered_data

    def get_technical_axiliary_data(self):
        """
        Acquires technical and economic indicators
        from alpha Vantage api

        Args:

        Returns:
            (pandas): pandas dataframe with the historical data
        """

        # Initialize an empty DataFrame
        pd_concat = pd.DataFrame()

        # Define indicator list to retrieve from alpha Vantage
        indicators = ["ATR", "ADX"]

        for indicator in indicators:
            # Query parameters
            params = {
                "function": indicator,
                "symbol": self.company,
                "apikey": self.alpha_api_key,
                "interval": "daily",
                "time_period": "14",
            }

            data = self.technical_data(params)
            pd_concat = data.loc[self.start_date : self.end_date].copy()

        # Define indicator list to retrieve from alpha Vantage
        indicators = ["SMA", "EMA", "RSI"]

        for indicator in indicators:
            # Query parameters
            params = {
                "function": indicator,
                "symbol": self.company,
                "apikey": self.alpha_api_key,
                "interval": "daily",
                "time_period": "14",
                "series_type": "close",
            }

            data = self.technical_data(params)
            pd_concat = pd.concat(
                [pd_concat, data.loc[self.start_date : self.end_date]], axis=1
            )

        # Define indicator list to retrieve from alpha Vantage
        indicators = ["OBV", "STOCH"]

        for indicator in indicators:
            # Query parameters
            params = {
                "function": indicator,
                "symbol": self.company,
                "apikey": self.alpha_api_key,
                "interval": "daily",
            }

            data = self.technical_data(params)
            pd_concat = pd.concat(
                [pd_concat, data.loc[self.start_date : self.end_date]], axis=1
            )

        return pd_concat
