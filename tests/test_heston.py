"""
Tests for the Heston calibration and the pricing that reads it back.

The market here is synthetic: quotes are generated from a known variance pair
under a known carry, so a fit that recovers them and a curve that reprices them
are both checkable to the cent, which no live chain allows.
"""

import unittest

import numpy as np
import pandas as pd

from base_alpha.models.heston import (
    STYLIZED_KAPPA,
    STYLIZED_RHO,
    STYLIZED_SIGMA,
    calibrate_v0_theta,
    estimate_heston_parameters,
    get_heston_fft_calls,
    get_heston_fft_puts,
    parity_carry,
)

TRUE_V0 = 0.045
TRUE_THETA = 0.030
# A forward well above spot and a discount well below one: the carry the hardcoded
# 3% cannot reproduce, which is the whole point of the regression below.
SPOT = 100.0
CARRY = {0.25: (0.982, 108.0), 0.75: (0.949, 121.0)}


def _quote(forward, discount, tau, strikes, is_call):
    """
    Prices contracts the way the calibration's own forward measure does.

    :param forward: Parity forward for the expiry.
    :param discount: Parity discount factor for the expiry.
    :param tau: Time to maturity in years.
    :param strikes: Strikes to price.
    :param is_call: True for calls, False for puts.
    :return: Prices as a numpy array.
    """
    _, calls = get_heston_fft_calls(
        S0=forward,
        v0=TRUE_V0,
        r=0.0,
        tau=tau,
        kappa=STYLIZED_KAPPA,
        theta=TRUE_THETA,
        sigma=STYLIZED_SIGMA,
        rho=STYLIZED_RHO,
        interpolate_strikes=np.asarray(strikes, dtype=float),
    )
    calls = discount * np.asarray(calls)
    if is_call:
        return calls
    return calls - discount * (forward - np.asarray(strikes, dtype=float))


def _maturities():
    """
    Builds a two-expiry calibration set of out-of-the-money quotes.

    :return: The list build_calibration_set would have produced for this market.
    """
    maturities = []
    for tau, (discount, forward) in CARRY.items():
        quotes = []
        for is_call in (True, False):
            strikes = (
                np.round(forward * np.linspace(1.02, 1.25, 8))
                if is_call
                else np.round(forward * np.linspace(0.75, 0.98, 8))
            )
            for strike, price in zip(strikes, _quote(forward, discount, tau, strikes, is_call)):
                # a flat spread: every quote carries the same weight in the fit
                quotes.append((float(strike), float(price), 0.5, is_call))
        maturities.append(
            {"tau": tau, "discount": discount, "forward": forward, "quotes": quotes}
        )
    return maturities


def _history():
    """
    A flat price history, so the realized fallback can never supply the variance.

    :return: An OHLCV frame with Close pinned at SPOT.
    """
    return pd.DataFrame({"Close": [SPOT] * 300})


class ParityCarryTest(unittest.TestCase):
    """Covers what parity_carry recovers and when it declines."""

    def test_recovers_the_carry_the_quotes_were_built_with(self):
        tau, (discount, forward) = 0.25, CARRY[0.25]
        strikes = np.round(forward * np.linspace(0.85, 1.15, 12))
        sides = {
            side: (
                strikes,
                _quote(forward, discount, tau, strikes, side == "calls"),
                np.full(len(strikes), 0.5),
            )
            for side in ("calls", "puts")
        }
        got_discount, got_forward = parity_carry(sides)
        self.assertAlmostEqual(got_discount, discount, places=4)
        self.assertAlmostEqual(got_forward, forward, places=2)

    def test_a_missing_side_gives_none(self):
        self.assertIsNone(parity_carry({"calls": (np.array([1.0]), np.array([1.0]), None)}))
        self.assertIsNone(parity_carry({}))


class CalibratedPricingTest(unittest.TestCase):
    """Covers that the curve shown is the curve that was fitted."""

    def setUp(self):
        self.maturities = _maturities()
        self.calibration = calibrate_v0_theta(self.maturities)
        self.assertIsNotNone(self.calibration, "the synthetic surface must calibrate")

    def test_the_fit_recovers_the_variance_pair(self):
        self.assertAlmostEqual(self.calibration["v0"], TRUE_V0, places=3)
        self.assertAlmostEqual(self.calibration["theta"], TRUE_THETA, places=3)

    def test_calibrated_curve_reprices_the_quotes_under_non_zero_carry(self):
        # The regression this file exists for. The fit runs in each expiry's
        # forward measure, so the panel's curve only goes through the quoted mids
        # if it is priced there too - with the spot and a nominal rate it does not.
        for tau, (discount, forward) in CARRY.items():
            params = estimate_heston_parameters(
                _history(),
                tau=tau,
                calibration=self.calibration,
                carry=(discount, forward),
            )
            quotes = [q for m in self.maturities if m["tau"] == tau for q in m["quotes"]]
            strikes = np.array([q[0] for q in quotes])
            _, calls = get_heston_fft_calls(**params, interpolate_strikes=strikes)
            _, puts = get_heston_fft_puts(**params, interpolate_strikes=strikes)
            shown = np.where([q[3] for q in quotes], calls, puts)
            mids = np.array([q[1] for q in quotes])
            self.assertLess(np.abs(shown - mids).max(), 0.02)

    def test_without_the_carry_the_curve_misses_the_quotes(self):
        # Guards the test above: it would pass on any pricing if the carry made no
        # difference. At this forward the nominal rate is off by dollars a contract.
        tau, (_, forward) = 0.25, CARRY[0.25]
        params = estimate_heston_parameters(
            _history(), tau=tau, calibration=self.calibration
        )
        quotes = [q for m in self.maturities if m["tau"] == tau for q in m["quotes"]]
        strikes = np.array([q[0] for q in quotes])
        _, calls = get_heston_fft_calls(**params, interpolate_strikes=strikes)
        _, puts = get_heston_fft_puts(**params, interpolate_strikes=strikes)
        shown = np.where([q[3] for q in quotes], calls, puts)
        mids = np.array([q[1] for q in quotes])
        self.assertGreater(np.abs(shown - mids).max(), 1.0)


class FallbackPricingTest(unittest.TestCase):
    """Covers that the two uncalibrated tiers are untouched by the carry."""

    def test_implied_tier_keeps_the_nominal_rate_and_the_spot(self):
        params = estimate_heston_parameters(_history(), market_v0=0.04, tau=0.25)
        self.assertEqual(params["r"], 0.03)
        self.assertEqual(params["S0"], SPOT)
        self.assertEqual(params["v0"], 0.04)
        self.assertEqual(params["theta"], 0.04)

    def test_a_carry_without_a_calibration_is_ignored(self):
        params = estimate_heston_parameters(
            _history(), market_v0=0.04, tau=0.25, carry=(0.949, 121.0)
        )
        self.assertEqual(params["r"], 0.03)
        self.assertEqual(params["S0"], SPOT)

    def test_realized_tier_keeps_the_nominal_rate_and_the_spot(self):
        history = pd.DataFrame({"Close": SPOT * np.exp(np.cumsum(
            np.random.default_rng(0).normal(0, 0.01, 300)
        ))})
        params = estimate_heston_parameters(history, tau=0.25)
        self.assertEqual(params["r"], 0.03)
        self.assertAlmostEqual(params["S0"], float(history["Close"].iloc[-1]))


if __name__ == "__main__":
    unittest.main()
