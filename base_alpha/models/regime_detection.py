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
        # Maps a reported regime back to the hidden state it came from, since
        # fit_predict may relabel them by volatility. Set during fit_predict.
        self._state_order = list(range(n_regimes))

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
            # transmat_ is left untouched by the relabelling, so anything reading it
            # has to go through this mapping or it will report the wrong state.
            self._state_order = [1, 0]
        else:
            self._state_order = [0, 1]

        df_res = df_ohlcv.loc[df_features.index].copy()
        df_res["Regime"] = regimes
        df_res["Prob_LowVola"] = probabilities[:, 0]
        df_res["Prob_HighVola"] = probabilities[:, 1]
        df_res["volatility"] = df_features["volatility"]

        return df_res

    def regime_summary(self, df_result: pd.DataFrame) -> dict:
        """
        Descriptive statistics the fitted HMM already holds but does not return.

        :param df_result: The DataFrame produced by fit_predict().
        :return: Dict with the current regime, how many days it has run, and the
            expected duration and annualised volatility of each regime.
        """
        regimes = df_result["Regime"].to_numpy()
        current = int(regimes[-1])

        # length of the run the series currently sits in
        run_length = 1
        while run_length < len(regimes) and regimes[-1 - run_length] == current:
            run_length += 1

        # The time spent in a state is geometric: staying with probability p_ii each
        # step gives a mean run of 1 / (1 - p_ii). _state_order undoes the relabelling.
        diagonal = np.diag(self.model.transmat_)[self._state_order]
        expected = [
            float(1.0 / (1.0 - p)) if p < 1.0 else float("inf") for p in diagonal
        ]

        log_return = np.log(df_result["Close"] / df_result["Close"].shift(1))
        vols = [
            float(log_return[regimes == i].std() * np.sqrt(252))
            for i in range(self.n_regimes)
        ]
        return {
            "current": current,
            "run_length": run_length,
            "expected_durations": expected,
            "annualised_vols": vols,
        }
