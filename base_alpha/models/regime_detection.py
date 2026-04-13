import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class RegimeDetector:
    """
    A class for detecting market regimes using a Hidden Markov Model (HMM).
    """

    def __init__(
        self,
        n_regimes=2,
        covariance_type="full",
        n_iter=1000,
        random_state=42,
        vola_window=20,
    ):
        """
        Initializes the RegimeDetector with the specified parameters.

        :param n_regimes: Number of hidden states, defaults to 2.
        :param covariance_type: Type of covariance matrix, defaults to 'full'.
        :param n_iter: Number of iterations, defaults to 1000.
        :param random_state: Random state for reproducibility, defaults to 42.
        :param vola_window: Window size for volatility calculation, defaults to 20.
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.vola_window = vola_window
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )

    def _prepare_hmm_features(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the features for the HMM model from the OHLCV DataFrame.

        :param df_ohlcv: DataFrame containing OHLCV data.
        :return: A numpy array of features for the HMM model.
        """
        df = df_ohlcv.copy()

        # Calculate log returns
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        # Calculate volatility
        df["volatility"] = df["log_return"].rolling(window=self.vola_window).std()

        return df.dropna()

    def fit_predict(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Trains the HMM and predicts the market regimes.

        :param df_ohlcv: DataFrame containing OHLCV data.
        :return: A pandas DataFrame with OHLCV data, predicted regimes and probabilities.
        """
        df_features = self._prepare_hmm_features(df_ohlcv)

        # Use log returns and volatility as features
        features = df_features[["log_return", "volatility"]].values
        self.model.fit(features)
        regimes = self.model.predict(features)
        probabilities = self.model.predict_proba(features)

        # ensure concistent labeling of regimes across different runs by sorting them based on
        # mean volatility
        vola_means = [features[regimes == i, 1].mean() for i in range(self.n_regimes)]
        # swap regimes if regime 0 has higher mean volatility than regime 1
        if vola_means[0] > vola_means[1]:
            regimes = 1 - regimes  # swap 0 and 1
            probabilities = probabilities[
                :, ::-1
            ]  # swap columns to match new regime labels

        df_res = df_ohlcv.loc[df_features.index].copy()
        df_res["Regime"] = regimes
        df_res["Prob_LowVola"] = probabilities[:, 0]
        df_res["Prob_HighVola"] = probabilities[:, 1]

        return df_res
