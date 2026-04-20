import numpy as np
import pandas as pd
import pandas_ta as ta
import xgboost as xgb


class Forecaster:
    """
    Alpha-Stack Forecaster (Level 1 Meta-Learner).
    Integrates HMM regimes, technical indicators, and (later) LSTM representations to predict
    future returns using XGBoost.
    """

    def __init__(self, forecast_horizon: int = 5, model_params: dict = None) -> None:
        """
        Initializes the Forecaster.

        :param forecast_horizon: Number of days to predict ahead.
        :param model_params: Dictionary of XGBoost hyperparameters.
        """
        self.forecast_horizon = forecast_horizon
        self.model_params = model_params or {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.05,
            "objective": "reg:squarederror",
            "random_state": 42,
        }
        self.model = xgb.XGBRegressor(**self.model_params)
        self.feature_cols = [
            "Regime",
            "Prob_LowVola",
            "Prob_HighVola",
            "RSI_14",
            "MACD_12_26_9",
            "MACDh_12_26_9",
            "MACDs_12_26_9",
            "dist_sma_200",
            "log_ret",
        ]

    def _prepare_features(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates technical indicators and aligns them with HMM regimes.

        :param df_ohlcv: DataFrame containing OHLCV and HMM regimes/probabilities.
        :return: DataFrame with added features and target.
        """
        df_features = df_ohlcv.copy()

        # 1. Technical Indicators (Explicit assignment)
        df_features["RSI_14"] = ta.rsi(df_features["Close"], length=14)

        macd_df = ta.macd(df_features["Close"], fast=12, slow=26, signal=9)
        df_features["MACD_12_26_9"] = macd_df["MACD_12_26_9"]
        df_features["MACDh_12_26_9"] = macd_df["MACDh_12_26_9"]
        df_features["MACDs_12_26_9"] = macd_df["MACDs_12_26_9"]

        # 2. Distance to Moving Average (Mean Reversion)
        sma_200 = ta.sma(df_features["Close"], length=200)
        df_features["dist_sma_200"] = (df_features["Close"] - sma_200) / sma_200

        # 3. Target Variable: Future cumulative log returns (Next N days)
        df_features["log_ret"] = np.log(
            df_features["Close"] / df_features["Close"].shift(1)
        )
        # Sum of the NEXT 'forecast_horizon' returns
        df_features["target"] = (
            df_features["log_ret"]
            .shift(-self.forecast_horizon)
            .rolling(window=self.forecast_horizon)
            .sum()
        )

        return df_features

    def train(self, df_with_regimes: pd.DataFrame) -> float:
        """
        Trains the XGBoost meta-learner.

        :param df_with_regimes: DataFrame from RegimeDetector.fit_predict().
        :return: The R^2 score of the model on the training set.
        """
        df_features = self._prepare_features(df_with_regimes)

        # Drop rows where we don't have enough data for indicators or target
        df_train = df_features.dropna(subset=self.feature_cols + ["target"])

        X = df_train[self.feature_cols]
        y = df_train["target"]

        self.model.fit(X, y)
        return float(self.model.score(X, y))

    def predict_latest(self, df_with_regimes: pd.DataFrame) -> float:
        """
        Predicts the return for the next N days based on the most recent data point.

        :param df_with_regimes: DataFrame with the latest data points.
        :return: Predicted cumulative log return.
        """
        df_features = self._prepare_features(df_with_regimes)

        # We only need the features for the very last row
        last_row = df_features.dropna(subset=self.feature_cols).iloc[[-1]]

        if last_row.empty:
            return 0.0

        prediction = self.model.predict(last_row[self.feature_cols])
        return float(prediction[0])
