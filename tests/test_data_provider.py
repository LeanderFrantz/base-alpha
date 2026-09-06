"""
Tests for YFProvider's quote validation.

yfinance is stubbed out throughout: these assert how a chain is read, not that
Yahoo is reachable, and the chains that matter here are the ones a live fetch
almost never returns.
"""

import unittest
from unittest.mock import patch

import pandas as pd

from base_alpha.data_provider.data_provider import YFProvider


def _chain(iv, bid, ask):
    """
    Builds a one-strike option chain, identical on both sides.

    :param iv: Implied volatility carried on the contract.
    :param bid: Bid, where 0 stands for a contract that is not quoted.
    :param ask: Ask, under the same convention.
    :return: An object with .calls and .puts, as yfinance's option_chain returns.
    """
    frame = pd.DataFrame(
        {"strike": [100.0], "impliedVolatility": [iv], "bid": [bid], "ask": [ask]}
    )
    return type("Chain", (), {"calls": frame, "puts": frame.copy()})()


class GetAtmIvTest(unittest.TestCase):
    """Covers which chains get_atm_iv accepts and which it refuses."""

    def _run(self, chain):
        """
        Calls get_atm_iv against a stubbed ticker with the spot at the strike.

        :param chain: The chain option_chain should return.
        :return: Whatever get_atm_iv returns.
        """
        with patch("base_alpha.data_provider.data_provider.yf.Ticker") as ticker:
            ticker.return_value.option_chain.return_value = chain
            ticker.return_value.history.return_value = pd.DataFrame(
                {"Close": [100.0]}
            )
            return YFProvider().get_atm_iv("TEST", "2026-09-18")

    def test_quoted_contract_yields_its_iv(self):
        self.assertAlmostEqual(self._run(_chain(0.25, 1.0, 1.2)), 0.25)

    def test_stale_iv_on_an_unquoted_contract_is_refused(self):
        # The case the realized volatility fallback exists for: the market is
        # shut, both sides are 0, and the plausible IV left on the contract is
        # whatever it was when it last traded. Returning it would price the
        # panel off a number no one is standing behind.
        self.assertIsNone(self._run(_chain(0.25, 0.0, 0.0)))

    def test_one_sided_quote_is_refused(self):
        self.assertIsNone(self._run(_chain(0.25, 1.0, 0.0)))
        self.assertIsNone(self._run(_chain(0.25, 0.0, 1.2)))

    def test_missing_quotes_are_refused(self):
        self.assertIsNone(self._run(_chain(0.25, float("nan"), float("nan"))))

    def test_iv_outside_bounds_is_still_refused_when_quoted(self):
        self.assertIsNone(self._run(_chain(9.0, 1.0, 1.2)))
        self.assertIsNone(self._run(_chain(0.0, 1.0, 1.2)))

    def test_missing_iv_on_a_quoted_contract_is_refused(self):
        self.assertIsNone(self._run(_chain(float("nan"), 1.0, 1.2)))


if __name__ == "__main__":
    unittest.main()
