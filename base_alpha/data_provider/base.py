from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    """
    Abstract base class for all data sources. Ensures that the system stays modular.
    """

    @abstractmethod
    def fetch_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical data for a given ticker and date range.

        :param ticker: The stock ticker symbol (e.g., 'AAPL').
        :param start_date: The start date for the data in 'YYYY-MM-DD' format.
        :param end_date: The end date for the data in 'YYYY-MM-DD'
        :return: A pandas DataFrame containing the historical data.
        """
        pass
