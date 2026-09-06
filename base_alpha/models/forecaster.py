import numpy as np
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Feature sets shared between the model and the dashboard's feature toggles.
# MANDATORY_FEATURES always feed the meta-learner; OPTIONAL_FEATURES can be
# switched off from the UI.
MANDATORY_FEATURES = ["Regime", "Prob_LowVola", "Prob_HighVola"]
OPTIONAL_FEATURES = ["RSI_14", "dist_sma_200", "MACD_12_26_9", "LSTM_Feature"]

# Longest indicator window used in _prepare_indicators. Shorter inputs cannot
# produce a single complete feature row.
SMA_WINDOW = 200


class LSTMModel(nn.Module):
    """
    A simple LSTM model for time-series forecasting.
    """

    def __init__(
        self, input_size: int, hidden_size: int = 32, num_layers: int = 1
    ) -> None:
        """
        Initializes the LSTM model.

        :param input_size: Number of features per time step.
        :param hidden_size: Number of hidden units in the LSTM layer.
        :param num_layers: Number of LSTM layers.
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param x: Input tensor of shape (batch, seq_len, input_size).
        :return: Output prediction tensor.
        """
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Take the last time step output for prediction
        out = self.fc(out[:, -1, :])
        return out


class Forecaster:
    """
    Alpha-Stack Forecaster.
    Level 0: LSTM (Deep Learning) + HMM (Statistical Regimes)
    Level 1: XGBoost (Meta-Learner)
    """

    def __init__(self, forecast_horizon: int = 5, lookback_window: int = 20) -> None:
        """
        Initializes the Forecaster.

        :param forecast_horizon: Number of days to predict ahead.
        :param lookback_window: Number of past days for LSTM sequences.
        """
        self.forecast_horizon = forecast_horizon
        self.lookback_window = lookback_window

        # Set torch seed for weight initialization
        torch.manual_seed(42)

        # Models
        self.lstm_model = None
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            objective="reg:squarederror",
            random_state=42,
        )

        # Utilities
        self.scaler = StandardScaler()
        # "LSTM_Feature" is the prediction from Level 0
        self.feature_cols_xgb = MANDATORY_FEATURES + OPTIONAL_FEATURES

    def _prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates technical indicators and prepares the target variable.

        :param df: Input DataFrame with OHLCV data.
        :return: DataFrame with added indicators and 'target'.
        """
        df = df.copy()
        df["RSI_14"] = ta.rsi(df["Close"], length=14)

        macd_df = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        if macd_df is not None:
            df["MACD_12_26_9"] = macd_df["MACD_12_26_9"].fillna(0.0)
        else:
            df["MACD_12_26_9"] = 0.0

        sma_200 = ta.sma(df["Close"], length=SMA_WINDOW)
        df["dist_sma_200"] = (df["Close"] - sma_200) / sma_200

        df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
        # Target: Sum of the NEXT 'forecast_horizon' returns
        df["target"] = (
            df["log_ret"]
            .shift(-self.forecast_horizon)
            .rolling(window=self.forecast_horizon)
            .sum()
        )
        return df

    def _create_sequences(
        self, data: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates sequences for LSTM training.

        :param data: Scaled features array.
        :param target: Target labels array.
        :return: Tuple of (sequences, targets) as numpy arrays.
        """
        X, y = [], []
        for i in range(len(data) - self.lookback_window):
            X.append(data[i : i + self.lookback_window])
            y.append(target[i + self.lookback_window])
        return np.array(X), np.array(y)

    def _train_lstm(self, df: pd.DataFrame, epochs: int = 20) -> None:
        """
        Level 0: Train LSTM to predict the target.

        :param df: Cleaned DataFrame with targets and features.
        :param epochs: Number of training epochs for the LSTM.
        """
        # Use OHLCV as raw inputs for LSTM
        raw_features = df[["Open", "High", "Low", "Close", "Volume"]].values
        target = df["target"].values

        # Scale
        scaled_features = self.scaler.fit_transform(raw_features)

        X, y = self._create_sequences(scaled_features, target)

        # Convert to Tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).view(-1, 1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.lstm_model = LSTMModel(input_size=5)
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.lstm_model.train()
        for _ in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def _get_lstm_features(self, df: pd.DataFrame, batch_size: int = 1024) -> np.ndarray:
        """
        Generate LSTM predictions as features for XGBoost.

        Every row from lookback_window onwards needs the prediction for the window
        ending just before it. The windows are stacked and run in batches rather
        than one forward pass per row.

        :param df: DataFrame containing historical data.
        :param batch_size: Rows per forward pass; bounds memory on long histories.
        :return: Numpy array of LSTM predictions (one for each row).
        """
        raw_features = df[["Open", "High", "Low", "Close", "Volume"]].values
        scaled_features = self.scaler.transform(raw_features)

        lstm_preds = np.full(len(df), np.nan)
        n_windows = len(df) - self.lookback_window
        if n_windows <= 0:
            return lstm_preds

        # sliding_window_view yields (n_windows + 1, n_features, lookback); drop the
        # trailing window, which would reach past the last row, and move the time
        # axis back into the middle so the shape matches (batch, seq_len, features).
        windows = np.lib.stride_tricks.sliding_window_view(
            scaled_features, self.lookback_window, axis=0
        )[:n_windows].transpose(0, 2, 1)

        self.lstm_model.eval()
        with torch.no_grad():
            for start in range(0, n_windows, batch_size):
                batch = np.ascontiguousarray(windows[start : start + batch_size])
                preds = self.lstm_model(torch.from_numpy(batch).float())
                offset = self.lookback_window + start
                lstm_preds[offset : offset + len(batch)] = preds.squeeze(-1).numpy()

        return lstm_preds

    def train(self, df_with_regimes: pd.DataFrame) -> float:
        """
        Trains both Level 0 (LSTM) and Level 1 (XGBoost).

        :param df_with_regimes: DataFrame from RegimeDetector.fit_predict().
        :return: The R^2 score of the XGBoost model on the training set.
        """
        # Prepare indicators and target
        df_features = self._prepare_indicators(df_with_regimes)
        df_clean = df_features.dropna(subset=["target", "RSI_14", "dist_sma_200"])

        # Train LSTM
        print("Training Level 0: LSTM...")
        self._train_lstm(df_clean)

        # Generate LSTM Features
        print("Generating Meta-Features...")
        lstm_preds = self._get_lstm_features(df_clean)
        df_meta = df_clean.copy()
        df_meta["LSTM_Feature"] = lstm_preds

        # Train XGBoost
        print("Training Level 1: XGBoost...")
        df_final = df_meta.dropna(subset=self.feature_cols_xgb)
        X = df_final[self.feature_cols_xgb]
        y = df_final["target"]

        self.xgb_model.fit(X, y)
        return float(self.xgb_model.score(X, y))

    def predict_latest(self, df_with_regimes: pd.DataFrame) -> float:
        """
        Predict using all features based on the most recent data point.

        :param df_with_regimes: DataFrame with the latest data points.
        :return: Predicted cumulative log return for the next N days.
        """
        df_features = self._prepare_indicators(df_with_regimes)

        # Generate LSTM Feature for latest window
        raw_features = df_features[["Open", "High", "Low", "Close", "Volume"]].values
        scaled_features = self.scaler.transform(raw_features)

        if len(scaled_features) < self.lookback_window:
            return 0.0

        last_window = scaled_features[-self.lookback_window :]
        window_tensor = torch.FloatTensor(last_window).unsqueeze(0)

        self.lstm_model.eval()
        with torch.no_grad():
            lstm_pred = self.lstm_model(window_tensor).item()

        # Prepare final features for XGBoost
        last_row = df_features.iloc[[-1]].copy()
        last_row["LSTM_Feature"] = lstm_pred

        X_final = last_row[self.feature_cols_xgb]
        prediction = self.xgb_model.predict(X_final)

        return float(prediction[0])
