import pandas as pd
import yfinance as yf

from .base import DataProvider


class YFProvider(DataProvider):
    """
    A data provider that fetches historical data using the yfinance library.
    """

    def fetch_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical OHLCV data for a given ticker and date range.

        :param ticker: The ticker symbol (e.g., "AAPL" for Apple Inc.).
        :param start_date: The start date for the data range.
        :param end_date: The end date for the data range.
        :return: A DataFrame containing the historical OHLCV data.
        """
        try:
            df_ohlcv = yf.download(
                ticker, start=start_date, end=end_date, auto_adjust=True
            )

            if df_ohlcv.empty:
                print(
                    f"No data found for ticker {ticker} between {start_date} and {end_date}."
                )
                return pd.DataFrame()

            if isinstance(df_ohlcv.columns, pd.MultiIndex):
                df_ohlcv.columns = df_ohlcv.columns.get_level_values(0)

            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            return df_ohlcv[required_columns]

        except Exception as e:
            print(f"An error occurred while fetching data for ticker {ticker}: {e}")
            return pd.DataFrame()
