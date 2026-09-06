import numpy as np
import pandas as pd
from scipy.fftpack import fft
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from typing import Dict

TRADING_DAYS_PER_YEAR = 252

# Shortest realized-volatility window. A one-day expiry would otherwise ask for a
# variance over a single return, which is undefined.
MIN_VOLA_WINDOW = 20


# Stylized parameters. Calibrating these too needs a global search and daily
# stability checks, and buys under one vol point on top of v0 and theta.
STYLIZED_KAPPA = 2.0
STYLIZED_SIGMA = 0.3
STYLIZED_RHO = -0.7

# Calibration guardrails. Anything the chain cannot satisfy makes the fit return
# None so the caller falls back instead of fitting noise.
MIN_PARITY_STRIKES = 6
# Parity is tightest near the money; far from it one leg is deep in the money with
# a wide spread and an unreliable mid.
PARITY_BAND = 0.15
MIN_QUOTES_PER_MATURITY = 6
MIN_CALIBRATION_MATURITIES = 2
MIN_CALIBRATION_QUOTES = 20
MIN_QUOTE_PRICE = 0.05
CALIBRATION_TAU_RANGE = (0.03, 1.0)
CALIBRATION_DAY_RANGE = (
    int(CALIBRATION_TAU_RANGE[0] * 365) + 1,
    int(CALIBRATION_TAU_RANGE[1] * 365) - 1,
)
CALIBRATION_MONEYNESS = (0.7, 1.3)
# Nominal rate the fallback tiers price with. The calibrated tier does not use it:
# it takes the carry the quotes imply instead.
FALLBACK_RATE = 0.03
VARIANCE_BOUNDS = (1e-4, 4.0)


def realized_vola_window(tau: float) -> int:
    """
    Number of trading days of history used to estimate variance for a given tenor.

    theta is pinned to v0 in estimate_heston_parameters, so the variance is flat
    over the option's life and v0 has to stand for that whole life rather than for
    the last month regardless of expiry.

    :param tau: Time to maturity in years.
    :return: Window length in trading days, at least MIN_VOLA_WINDOW.
    """
    return max(MIN_VOLA_WINDOW, int(round(tau * TRADING_DAYS_PER_YEAR)))


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


def _forward_and_discount(calls: tuple, puts: tuple) -> tuple[float, float] | None:
    """
    Implies the discount factor and the forward from put-call parity.

    C - P = D*F - D*K is linear in the strike, so regressing the call/put mid
    difference over the shared strikes yields both. That removes the interest rate
    and the dividend yield instead of assuming them, which matters: the parity
    forward on a live chain implies a carry well away from the hardcoded 3%.

    Two corrections keep it honest on real data. The regression runs only over
    strikes near the money, because parity is tight there and degrades where one
    leg is deep in the money on a wide spread. And the exchange lists American
    options, whose early exercise premium sits mostly in the puts and rises with
    the strike, which tilts the fitted slope and pushes the discount factor just
    above one. A discount factor above one is not possible, so it is clamped.

    :param calls: (strikes, mids, spreads) of the quoted calls.
    :param puts: The same for the puts.
    :return: (discount factor, forward), or None when the strikes barely overlap or
        the regression comes out implausible.
    """
    call_strikes, call_mids, _ = calls
    put_strikes, put_mids, _ = puts
    shared = np.intersect1d(call_strikes, put_strikes)
    if len(shared) < MIN_PARITY_STRIKES:
        return None

    call_lookup = dict(zip(call_strikes, call_mids))
    put_lookup = dict(zip(put_strikes, put_mids))
    difference = np.array([call_lookup[k] - put_lookup[k] for k in shared])

    # The strike where the two legs are worth the same sits closest to the forward,
    # which gives a band to regress over without needing the spot price.
    at_the_money = shared[np.argmin(np.abs(difference))]
    near = np.abs(shared / at_the_money - 1) <= PARITY_BAND
    if near.sum() < MIN_PARITY_STRIKES:
        near = np.ones_like(shared, dtype=bool)

    slope, intercept = np.polyfit(shared[near], difference[near], 1)
    if slope >= 0:
        return None

    discount = float(-slope)
    if not 0.85 < discount <= 1.05:
        return None
    forward = float(intercept / discount)
    return min(discount, 1.0), forward


def parity_carry(quotes: dict) -> tuple[float, float] | None:
    """
    The discount factor and forward one expiry's own quotes imply.

    The calibration is fitted in the forward measure, so a caller that wants to
    price it back has to discount the way _model_prices did. This hands it the
    same two numbers for the single expiry being shown.

    :param quotes: {"calls": (strikes, mids, spreads), "puts": ...} for one expiry,
        as YFProvider.get_option_quotes returns.
    :return: (discount factor, forward), or None when the chain cannot imply one -
        an unquoted side, or a regression the parity check rejects.
    """
    if "calls" not in quotes or "puts" not in quotes:
        return None
    return _forward_and_discount(quotes["calls"], quotes["puts"])


def build_calibration_set(chains: list, today: pd.Timestamp) -> list:
    """
    Turns raw quoted chains into the out-of-the-money set the fit runs on.

    Only OTM contracts are kept: they carry the liquidity, and including their ITM
    counterparts would count the same information twice through parity.

    :param chains: Entries of {"expiry", "calls", "puts"} with (strikes, mids,
        spreads) per side, as returned by YFProvider.get_calibration_quotes.
    :param today: Reference date for time to maturity.
    :return: One dict per usable maturity, with T, the discount factor, the forward
        and the surviving quotes. Empty when nothing qualifies.
    """
    maturities = []
    for chain in chains:
        tau = (pd.Timestamp(chain["expiry"]) - today).days / 365
        if not CALIBRATION_TAU_RANGE[0] < tau < CALIBRATION_TAU_RANGE[1]:
            continue
        if "calls" not in chain or "puts" not in chain:
            continue

        implied = _forward_and_discount(chain["calls"], chain["puts"])
        if implied is None:
            continue
        discount, forward = implied

        quotes = []
        for side, is_call in (("calls", True), ("puts", False)):
            strikes, mids, spreads = chain[side]
            out_of_money = strikes > forward if is_call else strikes < forward
            in_band = (strikes > CALIBRATION_MONEYNESS[0] * forward) & (
                strikes < CALIBRATION_MONEYNESS[1] * forward
            )
            keep = out_of_money & in_band & (mids > MIN_QUOTE_PRICE) & (spreads > 0)
            quotes.extend(
                (float(k), float(m), float(s), is_call)
                for k, m, s in zip(strikes[keep], mids[keep], spreads[keep])
            )

        if len(quotes) >= MIN_QUOTES_PER_MATURITY:
            maturities.append(
                {"tau": tau, "discount": discount, "forward": forward, "quotes": quotes}
            )
    return maturities


def _model_prices(maturities: list, v0: float, theta: float) -> list:
    """
    Prices every quote under the given variance pair, one FFT per maturity.

    Pricing with S0 set to the forward and r at zero puts the model in the forward
    measure, so the parity discount factor stays a separate multiplier rather than
    being conflated with the carry. Puts come from the model call by parity, so
    both sides rest on the same transform.
    """
    priced = []
    for maturity in maturities:
        strikes = np.array([q[0] for q in maturity["quotes"]], dtype=float)
        _, calls = get_heston_fft_calls(
            S0=maturity["forward"],
            v0=v0,
            r=0.0,
            tau=maturity["tau"],
            kappa=STYLIZED_KAPPA,
            theta=theta,
            sigma=STYLIZED_SIGMA,
            rho=STYLIZED_RHO,
            interpolate_strikes=strikes,
        )
        calls = maturity["discount"] * np.asarray(calls)
        priced.append(
            np.array(
                [
                    call if quote[3]
                    else call - maturity["discount"] * (maturity["forward"] - quote[0])
                    for call, quote in zip(calls, maturity["quotes"])
                ]
            )
        )
    return priced


def calibrate_v0_theta(maturities: list) -> Dict[str, float] | None:
    """
    Fits v0 and theta to quoted option prices, leaving kappa, sigma and rho fixed.

    Two parameters over several maturities is a well behaved problem: it converges
    to the same optimum from any starting point in well under a second, and needs
    no global search. Residuals are weighted by the bid-ask spread, since a sixty
    dollar contract and a fifty cent one carry the same information but not the
    same squared error.

    :param maturities: Output of build_calibration_set.
    :return: {"v0", "theta", "rmse_price", "n_quotes", "n_maturities"}, or None when
        there is too little quoted data. Outside US trading hours the whole chain
        is unquoted, so None is the normal answer then and callers must fall back.
    """
    total_quotes = sum(len(m["quotes"]) for m in maturities)
    if (
        len(maturities) < MIN_CALIBRATION_MATURITIES
        or total_quotes < MIN_CALIBRATION_QUOTES
    ):
        return None

    def residuals(params):
        stacked = []
        for maturity, priced in zip(maturities, _model_prices(maturities, *params)):
            mids = np.array([q[1] for q in maturity["quotes"]])
            spreads = np.array([q[2] for q in maturity["quotes"]])
            stacked.append((priced - mids) / spreads)
        return np.concatenate(stacked)

    try:
        fit = least_squares(
            residuals,
            x0=[0.05, 0.05],
            bounds=([VARIANCE_BOUNDS[0]] * 2, [VARIANCE_BOUNDS[1]] * 2),
            xtol=1e-10,
            ftol=1e-10,
        )
    except Exception as e:
        print(f"Heston calibration failed: {e}")
        return None
    if not fit.success:
        print(f"Heston calibration did not converge: {fit.message}")
        return None

    v0, theta = float(fit.x[0]), float(fit.x[1])
    # A parameter resting on its bound means the data did not pin it down - usually
    # too short a spread of maturities to see the long-run level at all.
    low, high = VARIANCE_BOUNDS
    if min(v0, theta) <= low * 10 or max(v0, theta) >= high * 0.99:
        print(f"Heston calibration hit a bound (v0={v0:.4f}, theta={theta:.4f}); discarding.")
        return None
    errors = np.concatenate(
        [
            priced - np.array([q[1] for q in maturity["quotes"]])
            for maturity, priced in zip(maturities, _model_prices(maturities, v0, theta))
        ]
    )
    return {
        "v0": v0,
        "theta": theta,
        "rmse_price": float(np.sqrt(np.mean(np.square(errors)))),
        "n_quotes": total_quotes,
        "n_maturities": len(maturities),
    }


def effective_vola(params: Dict[str, float]) -> float:
    """
    The volatility the model effectively prices its tenor with.

    Expected integrated variance over [0, tau] under Heston is
    theta + (v0 - theta) * (1 - exp(-kappa*tau)) / (kappa*tau). With theta pinned to
    v0 this is just v0, so the two fallback tiers are unaffected; once a calibration
    gives the two different values it is this, not v0, that drives the price.

    :param params: A dict from estimate_heston_parameters.
    :return: Annualised volatility.
    """
    v0, theta = params["v0"], params["theta"]
    kappa, tau = params["kappa"], params["tau"]
    if kappa * tau < 1e-8:
        return float(np.sqrt(v0))
    weight = (1.0 - np.exp(-kappa * tau)) / (kappa * tau)
    return float(np.sqrt(theta + (v0 - theta) * weight))


def estimate_heston_parameters(
    df_ohlcv: pd.DataFrame,
    market_v0: float = None,
    tau: float = 30 / 365,
    calibration: Dict[str, float] = None,
    carry: tuple[float, float] | None = None,
) -> Dict[str, float]:
    """
    Constructs Heston model parameters from the best source of variance available.

    Three tiers, in order: a calibration fitted to quoted option prices, the ATM
    implied variance with theta pinned to it, or realized variance from the price
    history. Only the last one always works, which is what keeps the model priced
    outside trading hours and for tickers with no listed options.

    :param df_ohlcv: OHLCV history, used for S0 and the realized fallback.
    :param market_v0: ATM implied variance, when one could be read.
    :param tau: Time to maturity in years.
    :param calibration: Result of calibrate_v0_theta, when a fit succeeded.
    :param carry: (discount factor, forward) from parity_carry for the expiry being
        priced. Used only alongside a calibration, which was fitted against them.
    :return: Parameter dict, holding exactly the keyword arguments the FFT pricer
        takes - callers splat it, so nothing else may be added to it.
    """
    S0 = float(df_ohlcv['Close'].iloc[-1])
    r = FALLBACK_RATE

    if calibration is not None:
        v0, theta = calibration["v0"], calibration["theta"]
        if carry is not None:
            # v0 and theta were fitted in the forward measure against the parity
            # discount, so pricing them off the spot at a nominal rate would draw a
            # curve that misses the very quotes it was fitted to. Setting S0 to F*D
            # and r to -ln(D)/tau puts the pricer back there exactly: its forward
            # comes out at F and its discount factor at D, and both the call
            # transform and the put parity below it follow. Doing it through the
            # two fields the pricer already takes keeps the dict a pure splat.
            discount, forward = carry
            S0 = forward * discount
            r = -float(np.log(discount)) / tau
    elif market_v0 is not None:
        # One expiry cannot separate v0 from theta, so theta stays pinned to it
        v0 = theta = market_v0
    else:
        log_returns = np.log(df_ohlcv['Close'] / df_ohlcv['Close'].shift(1)).dropna()
        window = realized_vola_window(tau)
        v0 = theta = float(log_returns.tail(window).var() * TRADING_DAYS_PER_YEAR)

    return {
        "S0": S0,
        "v0": v0,
        "theta": theta,
        "rho": STYLIZED_RHO,      # Leverage effect
        "kappa": STYLIZED_KAPPA,  # Stable reversion speed
        "sigma": STYLIZED_SIGMA,  # Vol-of-vol
        "r": r,
        "tau": tau,
    }
