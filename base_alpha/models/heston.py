import numpy as np
import pandas as pd
from scipy.fftpack import fft
from scipy.interpolate import interp1d
from typing import Dict


def heston_char_func(
    u: complex,
    S0: float,
    v0: float,
    r: float,
    tau: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
) -> complex:
    """
    Computes the characteristic function of the Heston model using the
    stable Albrecher-Gatheral formulation to avoid log-branch discontinuities.

    :param u: The transform variable (frequency).
    :param S0: Current asset price.
    :param v0: Initial variance.
    :param r: Risk-free interest rate.
    :param tau: Time to maturity (in years).
    :param kappa: Rate of mean reversion of the variance.
    :param theta: Long-term mean of the variance.
    :param sigma: Volatility of volatility (vol-of-vol).
    :param rho: Correlation between asset returns and variance.
    :return: Complex-valued characteristic function evaluated at u.
    """
    d = np.sqrt((rho * sigma * u * 1j - kappa) ** 2 + sigma**2 * (u * 1j + u**2))
    g = (kappa - rho * sigma * u * 1j - d) / (kappa - rho * sigma * u * 1j + d)

    C = r * u * 1j * tau + (kappa * theta / sigma**2) * (
        (kappa - rho * sigma * u * 1j - d) * tau
        - 2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g))
    )
    D = ((kappa - rho * sigma * u * 1j - d) / sigma**2) * (
        (1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau))
    )

    return np.exp(C + D * v0 + u * 1j * np.log(S0))


def get_heston_fft_calls(
    S0: float,
    v0: float,
    r: float,
    tau: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    interpolate_strikes: np.ndarray | None = None,
    n: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates European Call option prices using the Heston FFT approach.

    This function generates a dense grid of strikes and prices via Fast Fourier
    Transform and optionally interpolates them to specific user-defined strikes.

    :param S0: Current asset price.
    :param v0: Initial variance.
    :param r: Risk-free rate.
    :param tau: Time to maturity.
    :param kappa: Mean reversion speed.
    :param theta: Long-term variance.
    :param sigma: Vol-of-vol.
    :param rho: Price-vol correlation.
    :param interpolate_strikes: Array-like of specific strikes to interpolate for.
    :param n: Power of 2 determining the FFT grid size (N = 2^n).
    :return: Tuple of (strikes, call_prices).
    """
    N = 2**n
    upper_limit = 100
    alpha = 1.5
    delta_u = upper_limit / N
    u = np.arange(N) * delta_u

    lambda_grid = (2 * np.pi) / (N * delta_u)
    b = (N * lambda_grid) / 2
    k_strikes = -b + np.arange(N) * lambda_grid

    u_mod = u - (alpha + 1) * 1j
    phi = heston_char_func(u_mod, S0, v0, r, tau, kappa, theta, sigma, rho)

    # Integration & FFT
    numerator = np.exp(-r * tau) * phi
    denominator = (alpha + 1j * u) * (alpha + 1 + 1j * u)
    weights = (np.ones(N) * (3 + (-1) ** np.arange(1, N + 1))) / 3
    weights[0] = 1 / 3

    integrand = (numerator / denominator) * np.exp(1j * u * b) * delta_u * weights
    fft_values = fft(integrand).real

    call_prices = (np.exp(-alpha * k_strikes) / np.pi) * fft_values
    strikes = np.exp(k_strikes)

    # filter for reasonable strikes, to prevent unstable prices around 0
    mask = (strikes > S0 * 0.1) & (strikes < S0 * 3.0)
    strikes = strikes[mask]
    call_prices = call_prices[mask]

    if interpolate_strikes is not None:
        f_interp = interp1d(
            strikes, call_prices, kind="cubic", fill_value="extrapolate"
        )
        strikes = interpolate_strikes
        call_prices = f_interp(interpolate_strikes)

    return strikes, np.maximum(0, call_prices)


def get_heston_fft_puts(
    S0: float,
    v0: float,
    r: float,
    tau: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    interpolate_strikes: np.ndarray = None,
    n: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates European Put option prices using the Heston FFT approach.

    :param S0: Current asset price.
    :param v0: Initial variance.
    :param r: Risk-free rate.
    :param tau: Time to maturity.
    :param kappa: Mean reversion speed.
    :param theta: Long-term variance.
    :param sigma: Vol-of-vol.
    :param rho: Price-vol correlation.
    :param interpolate_strikes: Array-like of specific strikes to interpolate for.
    :param n: Power of 2 determining the FFT grid size.
    :return: Tuple of (strikes, put_prices).
    """
    strikes, call_prices = get_heston_fft_calls(
        S0, v0, r, tau, kappa, theta, sigma, rho, interpolate_strikes, n
    )

    # Vectorized calculation for all strikes in the grid
    put_prices = call_prices - S0 + strikes * np.exp(-r * tau)

    # ensure no negative prices
    return strikes, np.maximum(0, put_prices)


def estimate_heston_parameters(df_ohlcv: pd.DataFrame, market_v0: float = None, tau: float = 30 / 365) -> Dict[str, float]:
    """
    Constructs Heston model parameters using dynamic S0, (optionally) provided market v0/theta, and fixed stylized parameters.
    """
    S0 = float(df_ohlcv['Close'].iloc[-1])

    # If market_v0 is provided (from IV), use it, otherwise estimate historical v0
    if market_v0 is None:
        log_returns = np.log(df_ohlcv['Close'] / df_ohlcv['Close'].shift(1)).dropna()
        v0 = float(log_returns.tail(20).var() * 252)
    else:
        v0 = market_v0

    # Stylized parameters
    return {
        "S0": S0,
        "v0": v0,
        "theta": v0,   # Prevent immediate mean-reversion drift
        "rho": -0.7,    # Leverage effect
        "kappa": 2.0,   # Stable reversion speed
        "sigma": 0.3,   # Vol-of-vol
        "r": 0.03,
        "tau": tau,
    }
