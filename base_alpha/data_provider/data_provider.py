import numpy as np
import pandas as pd
import yfinance as yf

from .base import DataProvider

# Yahoo reports a 1e-5 placeholder when it has no implied volatility, and its
# solver bottoms out on a ladder of tiny values when the chain carries no quotes
# (outside US trading hours the whole chain is regularly unquoted). Anything
# outside this band is not a market price and must not reach the pricing model.
IV_BOUNDS = (0.01, 3.0)


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
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
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
    def get_atm_iv(self, ticker: str, target_days: int) -> float | None:
        """
        Fetches ATM Implied Volatility for the expiry closest to target_days.
        Averages ATM Call and Put IVs.

        :param ticker: The ticker symbol.
        :param target_days: Desired days to expiry; the closest listed one is used.
        :return: The averaged ATM implied volatility, or None when the chain cannot
            supply a usable one - no options listed (Yahoo carries none for European
            tickers), no implied volatility, or a value outside IV_BOUNDS. Callers
            are expected to fall back rather than receive an invented number.
        """
        try:
            t = yf.Ticker(ticker)
            expirations = pd.to_datetime(t.options)
            if expirations.empty:
                print(f"No option chain listed for {ticker}.")
                return None

            # Find closest expiration
            target_date = pd.Timestamp.now() + pd.Timedelta(days=target_days)
            # Use .values.astype(np.int64) to get nanoseconds for comparison
            diffs = (expirations - target_date).values
            closest_exp = expirations[np.abs(diffs).argmin()]
            exp_str = closest_exp.strftime("%Y-%m-%d")

            chain = t.option_chain(exp_str)
            spot = t.history(period="1d")["Close"].iloc[-1]

            # ATM Strike (closest to spot)
            atm_strike_call = (chain.calls["strike"] - spot).abs().idxmin()
            atm_strike_put = (chain.puts["strike"] - spot).abs().idxmin()

            # Get IVs
            call_iv = chain.calls.loc[atm_strike_call, "impliedVolatility"]
            put_iv = chain.puts.loc[atm_strike_put, "impliedVolatility"]

            if pd.isna(call_iv) or pd.isna(put_iv):
                print(f"No implied volatility quoted for {ticker} at {exp_str}.")
                return None

            atm_iv = float((call_iv + put_iv) / 2)
            low, high = IV_BOUNDS
            if not low <= atm_iv <= high:
                print(
                    f"Discarding implausible ATM IV for {ticker}: {atm_iv:.6f} "
                    f"(outside {low:.0%}-{high:.0%}); the chain is likely unquoted."
                )
                return None

            return atm_iv
        except Exception as e:
            print(f"Error fetching IV: {e}")
            return None
