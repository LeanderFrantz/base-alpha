import io
import os
import threading
import dash
from dash import dcc, html, Input, Output, State, exceptions
import dash_bootstrap_components as dbc
from base_alpha.data_provider.data_provider import YFProvider
from base_alpha.models.regime_detection import RegimeDetector
from base_alpha.models.forecaster import (
    Forecaster,
    MANDATORY_FEATURES,
    OPTIONAL_FEATURES,
    SMA_WINDOW,
)
from base_alpha.models.heston import (
    get_heston_fft_calls,
    get_heston_fft_puts,
    estimate_heston_parameters,
    realized_vola_window,
    build_calibration_set,
    calibrate_v0_theta,
    effective_vola,
    CALIBRATION_DAY_RANGE,
)
from base_alpha.analytics.visualizer import Visualizer
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# Trading days the Forecaster is trained on. The horizon slider only rescales
# this prediction, it does not retrain the model on a different horizon.
TRAIN_HORIZON = 5

# Selectable confidence levels for the forecast band, plus the narrow inner band
# that is always drawn to give the projection its fan shape.
CI_LEVELS = [68, 90, 95]
INNER_CI_LEVEL = 50

# Trading days of history shown in the default chart view.
DEFAULT_LOOKBACK = 120

# Expiry the dropdown preselects, and the tenor priced when a ticker lists no
# options at all.
DEFAULT_EXPIRY_DAYS = 30

# Dropdown value meaning "this ticker lists no expiries". Distinct from None,
# which only means the dropdown has not been filled yet.
NO_EXPIRY = "__none__"

# Upper bound for the module level caches below.
MAX_CACHE_ENTRIES = 8

# How long a quoted price or implied volatility may be served from cache. Both
# move while the market is open, and their keys otherwise hold nothing that
# changes intraday, so without this they would be frozen for the whole process.
QUOTE_TTL_SECONDS = 300

# Header colours. Amber flags a metric that is not a market quote but a fallback,
# so a substituted number can never pass for a real one.
METRIC_COLOR = "#00d1b2"
LABEL_COLOR = "#6c757d"
FALLBACK_COLOR = "#f0ad4e"
UP_COLOR = "#2ecc71"

# Header type scale. The detail line carries data, so it is not smaller than the
# caption above it.
CAPTION_SIZE = "0.8rem"
DETAIL_SIZE = "0.85rem"
DOWN_COLOR = "#ff5050"

# Calendar days the data may lag before it is flagged as stale. A weekend plus a
# public holiday accounts for four.
STALE_AFTER_DAYS = 4

# Shortest history the feature pipeline can produce a complete row from.
MIN_HISTORY = SMA_WINDOW + TRAIN_HORIZON

# Default dates: Today - 5 years to Today
end_date = datetime.now() + timedelta(
    days=1
)  # Adjusted to ensure we have data for today
start_date = end_date - timedelta(days=5 * 365)

def _metric_col(
    label_text: str,
    value_id: str,
    label_id: str | None = None,
    detail_id: str | None = None,
    tooltip: str | None = None,
) -> dbc.Col:
    """
    One tile of the metrics header: a grey caption, the value, and an optional
    smaller detail line beneath it.

    :param label_text: Caption shown initially.
    :param value_id: Component id of the value.
    :param label_id: Component id of the caption, when a callback rewrites it.
    :param detail_id: Component id of the detail line, when the tile has one.
    :param tooltip: Static hover text explaining the figure. Tiles whose explanation
        depends on the data carry a Tooltip of their own instead.
    :return: An auto-width column, so the row divides evenly however many there are.
    """
    caption = html.Small(
        label_text, style={"color": LABEL_COLOR, "fontSize": CAPTION_SIZE}
    )
    if label_id:
        caption.id = label_id

    children = [caption, html.H5(id=value_id, children="-", style={"color": METRIC_COLOR})]
    if detail_id:
        children.append(
            html.Small(id=detail_id, children="", style={"fontSize": DETAIL_SIZE})
        )
    if tooltip:
        children.append(dbc.Tooltip(tooltip, target=value_id, placement="bottom"))
    return dbc.Col(html.Div(children), width=True)


app.layout = dbc.Container(
    [
        # data stores for caching
        dcc.Store(id="data-store"),
        dbc.Row(
            [
                dbc.Col(
                    html.H2(
                        "BASE-ALPHA | Performance Terminal",
                        className="text-primary mt-3",
                    ),
                    width=12,
                )
            ],
            className="mb-4 border-bottom border-primary pb-3",
        ),
        dbc.Alert(
            id="status-alert",
            color="danger",
            is_open=False,
            dismissable=True,
            className="mb-3",
        ),
        dbc.Alert(
            id="option-alert",
            color="danger",
            is_open=False,
            dismissable=True,
            className="mb-3",
        ),
        dbc.Tooltip(id="iv-tooltip", target="metric-iv", placement="bottom"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Data Settings"),
                                dbc.CardBody(
                                    [
                                        html.Label("Ticker:"),
                                        dbc.Input(
                                            id="ticker-input",
                                            value="AAPL",
                                            type="text",
                                            # Enter submits, same as the button
                                            n_submit=0,
                                            className="mb-3",
                                        ),
                                        html.Label("Date Range:"),
                                        dcc.DatePickerRange(
                                            id="date-picker",
                                            start_date=start_date.strftime("%Y-%m-%d"),
                                            end_date=end_date.strftime("%Y-%m-%d"),
                                            className="mb-3",
                                        ),
                                        dbc.Button(
                                            "Fetch New Data",
                                            id="fetch-btn",
                                            color="warning",
                                            className="w-100",
                                        ),
                                        html.Label(
                                            "Forecast Horizon (Days):", className="mt-3"
                                        ),
                                        dcc.Slider(
                                            id="horizon-slider",
                                            min=1,
                                            max=30,
                                            step=1,
                                            value=5,
                                            marks={i: str(i) for i in range(5, 31, 5)},
                                        ),
                                        html.Label(
                                            "HMM Volatility Lookback (Days):",
                                            className="mt-3",
                                        ),
                                        dcc.Slider(
                                            id="vola-slider",
                                            min=10,
                                            max=100,
                                            step=5,
                                            value=20,
                                            marks={
                                                i: str(i) for i in range(10, 101, 20)
                                            },
                                        ),
                                        html.Label(
                                            "Option Expiry:", className="mt-3"
                                        ),
                                        dcc.Dropdown(
                                            id="expiry-dropdown",
                                            options=[],
                                            value=None,
                                            clearable=False,
                                            placeholder="Loading expiries...",
                                            # The CYBORG theme leaves the menu on a
                                            # light background, so force dark text.
                                            style={"color": "#000000"},
                                        ),
                                        html.Label(
                                            "Forecast Confidence:", className="mt-3"
                                        ),
                                        dcc.Dropdown(
                                            id="ci-level",
                                            options=[
                                                {"label": f"{lvl}%", "value": lvl}
                                                for lvl in CI_LEVELS
                                            ],
                                            value=90,
                                            clearable=False,
                                            # The CYBORG theme leaves the menu on a
                                            # light background, so force dark text.
                                            style={"color": "#000000"},
                                        ),
                                    ]
                                ),
                                dbc.CardHeader("Predictive Feature Selection"),
                                dbc.CardBody(
                                    [
                                        dbc.Checklist(
                                            id="feature-toggle-all",
                                            options=[
                                                {
                                                    "label": "Enable/Disable All",
                                                    "value": "all",
                                                }
                                            ],
                                            value=["all"],
                                            inline=True,
                                            className="mb-2",
                                        ),
                                        dbc.Checklist(
                                            id="feature-toggles",
                                            options=[
                                                {"label": name, "value": name}
                                                for name in OPTIONAL_FEATURES
                                            ],
                                            value=list(OPTIONAL_FEATURES),
                                            className="mb-2",
                                        ),
                                    ]
                                ),
                            ],
                            color="dark",
                            outline=True,
                        )
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        # Metrics Bar
                        dbc.Card(
                            dbc.CardBody(
                                dbc.Row(
                                    [
                                        _metric_col(
                                            "ACTIVE TICKER",
                                            "metric-ticker",
                                            tooltip=(
                                                "The symbol currently loaded, and the date of the most recent bar beneath it. Everything on this page is computed from data up to that date, so the figure turns amber once the series falls behind."
                                            ),
                                            detail_id="metric-asof",
                                        ),
                                        _metric_col(
                                            "LAST PRICE",
                                            "metric-price",
                                            tooltip=(
                                                "Close of the most recent bar and its change against the previous close. This is not a live quote: it only moves when you fetch data again."
                                            ),
                                            detail_id="metric-change",
                                        ),
                                        _metric_col(
                                            "EXP. MOVE",
                                            "metric-move",
                                            tooltip=(
                                                "Forecast of the stacked model - an LSTM feeding XGBoost alongside the HMM regime probabilities and technical indicators. It is trained on a fixed 5 trading day horizon and rescaled linearly to the horizon you select, so long horizons are an extrapolation rather than a separate model. The line beneath is the confidence band at that horizon, as a range around the current price."
                                            ),
                                            label_id="metric-move-label",
                                            detail_id="metric-move-band",
                                        ),
                                        _metric_col(
                                            "CURRENT REGIME",
                                            "metric-regime",
                                            tooltip=(
                                                "Hidden Markov state fitted on log returns and rolling volatility. The detail line reads: how long the current state has lasted, the duration expected from the model's transition matrix, and the volatility actually realised in this state, annualised."
                                            ),
                                            detail_id="metric-regime-detail",
                                        ),
                                        _metric_col(
                                            "FAIR ATM CALL (30D)",
                                            "metric-call",
                                            tooltip=(
                                                "Heston price of the call struck nearest spot for the selected expiry, from the FFT pricer. The panel below plots the whole curve against the quoted market prices."
                                            ),
                                            label_id="label-call",
                                        ),
                                        _metric_col(
                                            "FAIR ATM PUT (30D)",
                                            "metric-put",
                                            tooltip=(
                                                "Heston price of the put struck nearest spot for the selected expiry, from the FFT pricer. The panel below plots the whole curve against the quoted market prices."
                                            ),
                                            label_id="label-put",
                                        ),
                                        _metric_col(
                                            "IMPLIED VOLA (30D)",
                                            "metric-iv",
                                            label_id="label-iv",
                                        ),
                                    ],
                                    className="text-center",
                                ),
                            ),
                            style={
                                "backgroundColor": "#0d1117",
                                "border": "1px solid #00d1b2",
                                "borderRadius": "10px",
                            },
                            className="mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="main-chart", style={"height": "75vh"}
                                        )
                                    ),
                                    width=12,
                                ),
                            ],
                            className="g-0",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="heston-chart",
                                            style={"height": "40vh"},
                                        )
                                    ),
                                    width=12,
                                    className="mt-3",
                                ),
                            ],
                        ),
                    ],
                    width=9,
                ),
            ]
        ),
    ],
    fluid=True,
)


# Rescales the price axis to whatever x-window is on screen. Plotly only offers
# "fit everything" or a fixed range, so panning or zooming would otherwise leave
# the chart scaled to the window it was built with. Runs in the browser, so it
# costs no callback round trip and never retrains a model.
Y_RESCALE_JS = """
function(relayoutData, figure) {
    var dc = window.dash_clientside;
    if (!figure || !relayoutData) { return dc.no_update; }

    var keys = Object.keys(relayoutData);
    var touchesX = keys.some(function(k) {
        return k.indexOf('xaxis') === 0 &&
               (k.indexOf('.range') > 0 || k.indexOf('.autorange') > 0);
    });
    if (!touchesX) { return dc.no_update; }

    // Plotly 6 ships numeric columns as base64 typed arrays ({dtype, bdata}),
    // so trace.y has to be decoded before it can be read. Date columns arrive as
    // plain string arrays.
    function toArray(column) {
        if (!column) { return null; }
        if (Array.isArray(column)) { return column; }
        if (column.bdata === undefined) { return null; }
        var binary = atob(column.bdata);
        var buffer = new ArrayBuffer(binary.length);
        var bytes = new Uint8Array(buffer);
        for (var i = 0; i < binary.length; i++) { bytes[i] = binary.charCodeAt(i); }
        switch (column.dtype) {
            case 'f8': return new Float64Array(buffer);
            case 'f4': return new Float32Array(buffer);
            case 'i4': return new Int32Array(buffer);
            case 'i2': return new Int16Array(buffer);
            case 'i1': return new Int8Array(buffer);
            case 'u4': return new Uint32Array(buffer);
            case 'u2': return new Uint16Array(buffer);
            case 'u1': return bytes;
            default: return null;
        }
    }

    // Plotly hands dates back as "2026-03-10 12:00:00" or ISO, sometimes as epoch ms.
    function toMs(value) {
        if (value === undefined || value === null) { return null; }
        if (typeof value === 'number') { return value; }
        return new Date(String(value).replace(' ', 'T')).getTime();
    }

    var lo = relayoutData['xaxis2.range[0]'];
    var hi = relayoutData['xaxis2.range[1]'];
    if (lo === undefined) { lo = relayoutData['xaxis.range[0]']; }
    if (hi === undefined) { hi = relayoutData['xaxis.range[1]']; }
    // An autorange reset leaves no bounds, which means "show everything".
    var t0 = (lo === undefined) ? -Infinity : toMs(lo);
    var t1 = (hi === undefined) ? Infinity : toMs(hi);

    var lowest = Infinity;
    var highest = -Infinity;
    for (var d = 0; d < figure.data.length; d++) {
        var trace = figure.data[d];
        // Row 2 holds the regime probability on its own axis, fixed to 0..1.
        if ((trace.yaxis || 'y') !== 'y') { continue; }
        var xs = toArray(trace.x);
        var ys = toArray(trace.y);
        if (!xs || !ys) { continue; }
        for (var i = 0; i < xs.length; i++) {
            var t = toMs(xs[i]);
            if (t < t0 || t > t1) { continue; }
            var v = ys[i];
            if (v === null || v === undefined || !isFinite(v)) { continue; }
            if (v < lowest) { lowest = v; }
            if (v > highest) { highest = v; }
        }
    }
    if (!isFinite(lowest) || !isFinite(highest) || highest <= lowest) {
        return dc.no_update;
    }

    var pad = (highest - lowest) * 0.08;
    var next = [lowest - pad, highest + pad];
    var current = (figure.layout.yaxis || {}).range;
    // Bailing out when nothing changes also stops this from feeding itself.
    if (current &&
        Math.abs(current[0] - next[0]) < 1e-6 &&
        Math.abs(current[1] - next[1]) < 1e-6) {
        return dc.no_update;
    }

    var layout = Object.assign({}, figure.layout);
    layout.yaxis = Object.assign({}, layout.yaxis, {range: next, autorange: false});
    return Object.assign({}, figure, {layout: layout});
}
"""

# CALLBACK: Feature Toggle Logic
@app.callback(
    Output("feature-toggles", "value"),
    Input("feature-toggle-all", "value"),
    State("feature-toggles", "options"),
)
def toggle_all_features(all_selected, options):
    if "all" in (all_selected or []):
        return [opt["value"] for opt in options]
    return []


# CALLBACK: Fetch Data (Cached)
@app.callback(
    [
        Output("data-store", "data"),
        Output("status-alert", "children"),
        Output("status-alert", "color"),
        Output("status-alert", "is_open"),
    ],
    [Input("fetch-btn", "n_clicks"), Input("ticker-input", "n_submit")],
    [
        State("ticker-input", "value"),
        State("date-picker", "start_date"),
        State("date-picker", "end_date"),
    ],
)
def fetch_data(n_clicks, n_submit, ticker, start, end):

    print(f"DEBUG: Fetching data for {ticker}...")
    provider = YFProvider()
    df_ohlcv = provider.fetch_data(ticker, start, end)
    if df_ohlcv.empty:
        return (
            dash.no_update,
            f"No data returned for '{ticker}' between {start} and {end}. "
            "Check the ticker symbol and the date range.",
            "danger",
            True,
        )
    # to JSON for storage in dcc.Store
    return df_ohlcv.to_json(date_format="iso", orient="split"), "", "danger", False


# CALLBACK: Fill the expiry dropdown with the dates actually listed for the ticker
@app.callback(
    [
        Output("expiry-dropdown", "options"),
        Output("expiry-dropdown", "value"),
        Output("expiry-dropdown", "disabled"),
        Output("expiry-dropdown", "placeholder"),
    ],
    [Input("fetch-btn", "n_clicks"), Input("ticker-input", "n_submit")],
    State("ticker-input", "value"),
)
def load_expirations(n_clicks, n_submit, ticker):
    expirations = YFProvider().get_expirations(ticker)
    if not expirations:
        return [], NO_EXPIRY, True, "no listed expiries"

    today = pd.Timestamp.now().normalize()
    options = [
        {"label": f"{exp}  ({(pd.Timestamp(exp) - today).days}d)", "value": exp}
        for exp in expirations
    ]
    closest = min(
        expirations,
        key=lambda exp: abs((pd.Timestamp(exp) - today).days - DEFAULT_EXPIRY_DAYS),
    )
    return options, closest, False, ""


# Global cache for the forecaster model
model_cache = {}

# Global cache for Heston parameters
heston_cache = {}

# Global cache for the fitted surface, one entry per ticker per trading day.
calibration_cache = {}

# Global cache for quoted option prices, so the panel does not refetch the same
# chain the implied volatility already came from.
quotes_cache = {}

# Global cache for the HMM fit. Both panels need the same regimes, and they do not
# depend on horizon, expiry or confidence level, so those must not refit the model.
regime_cache = {}

# Guards the fit in _get_regimes. Dash answers callbacks on a thread pool, so both
# panels can miss the cache on the same key at once and fit the same HMM twice.
regime_lock = threading.Lock()


def _quote_bucket() -> int:
    """
    Returns the index of the interval the current quote fetch belongs to.

    Rolling the index into a cache key expires quoted market data without a
    separate sweep, and because the index counts from the epoch it also turns
    over at every trading day boundary.

    :return: Bucket index, which advances once per QUOTE_TTL_SECONDS.
    """
    return int(pd.Timestamp.now().timestamp() // QUOTE_TTL_SECONDS)


def _cache_store(cache: dict, key: str, value) -> None:
    """
    Stores a value in a module level cache and evicts the oldest entries once it
    grows past MAX_CACHE_ENTRIES. Without the bound, every slider position would
    leave a trained LSTM and XGBoost model behind for the lifetime of the process.

    :param cache: The cache dictionary to write to.
    :param key: Cache key.
    :param value: Value to store.
    """
    cache[key] = value
    while len(cache) > MAX_CACHE_ENTRIES:
        cache.pop(next(iter(cache)))


def _get_regimes(json_data: str, vola_val: int) -> tuple[pd.DataFrame, str, dict]:
    """
    Runs regime detection on the stored data, memoized on payload and window.

    :param json_data: Serialized OHLCV DataFrame from the data store.
    :param vola_val: HMM volatility lookback window in trading days.
    :return: Tuple of (DataFrame with regimes, dataset identifier, regime summary).
    """
    cache_key = f"{hash(json_data)}_{vola_val}"
    if cache_key in regime_cache:
        return regime_cache[cache_key]

    with regime_lock:
        # the callback we queued behind may have just fitted this very key
        if cache_key in regime_cache:
            return regime_cache[cache_key]

        df_ohlcv = pd.read_json(io.StringIO(json_data), orient="split")
        if len(df_ohlcv) < MIN_HISTORY:
            raise ValueError(
                f"only {len(df_ohlcv)} trading days loaded, but the {SMA_WINDOW}-day SMA "
                f"feature needs at least {MIN_HISTORY}. Widen the date range and fetch again."
            )

        detector = RegimeDetector(n_regimes=2, vola_window=vola_val)
        df_result = detector.fit_predict(df_ohlcv)
        data_id = f"{len(df_ohlcv)}_{df_ohlcv.index[0]}_{df_ohlcv.index[-1]}"
        # computed while the fitted detector is still around; transmat_ is otherwise lost
        summary = detector.regime_summary(df_result)
        _cache_store(regime_cache, cache_key, (df_result, data_id, summary))
        return df_result, data_id, summary


def _get_forecaster(
    df_result: pd.DataFrame,
    ticker: str,
    vola_val: int,
    selected_features: list,
    data_id: str,
) -> Forecaster:
    """
    Returns a trained forecaster, training one only when no cached model matches.

    :param df_result: DataFrame from RegimeDetector.fit_predict().
    :param ticker: Active ticker symbol.
    :param vola_val: HMM volatility lookback window, which shapes the regimes.
    :param selected_features: Optional features enabled in the UI.
    :param data_id: Stable identifier of the loaded dataset.
    :return: The trained Forecaster.
    """
    cache_key = f"{ticker}_{vola_val}_{str(selected_features)}_{data_id}"
    if cache_key not in model_cache:
        forecaster = Forecaster(forecast_horizon=TRAIN_HORIZON)
        forecaster.feature_cols_xgb = MANDATORY_FEATURES + list(selected_features or [])
        forecaster.train(df_result)
        _cache_store(model_cache, cache_key, forecaster)
    return model_cache[cache_key]


def _get_calibration(ticker: str) -> dict | None:
    """
    Fits v0 and theta to the quoted surface, once per ticker per trading day.

    :param ticker: The ticker symbol.
    :return: The calibration, or None whenever the chain cannot support one - the
        surface is unquoted outside US trading hours and absent entirely for the
        many tickers Yahoo lists no options for. Callers must fall back.
    """
    today = pd.Timestamp.now().normalize()
    cache_key = f"{ticker}_{today.date()}"
    if cache_key not in calibration_cache:
        chains = YFProvider().get_calibration_quotes(ticker, *CALIBRATION_DAY_RANGE)
        fit = calibrate_v0_theta(build_calibration_set(chains, today))
        print(f"DEBUG: calibration for {ticker}: {fit}")
        _cache_store(calibration_cache, cache_key, fit)
    return calibration_cache[cache_key]


def _get_heston(
    df_result: pd.DataFrame,
    ticker: str,
    expiry_date: str | None,
    expiry_days: int,
    vola_val: int,
    data_id: str,
) -> tuple[float | None, dict]:
    """
    Returns ATM implied volatility and Heston parameters for one listed expiry.

    The parameters are derived from df_result, so the key has to cover every input
    that changes it - ticker and expiry alone would serve stale values after the
    volatility slider or the date range moved. It also carries the quote bucket,
    because the implied volatility below is a live quote. The calibration behind
    it is keyed by trading day instead, so an intraday refresh reuses that fit.

    :param expiry_date: A listed expiry, or None when the ticker has none.
    :param expiry_days: Calendar days to that expiry, which sets tau.
    :return: (ATM implied volatility or None, Heston parameters, calibration or None).
    """
    cache_key = (
        f"{ticker}_{expiry_date}_{expiry_days}_{vola_val}_{data_id}_{_quote_bucket()}"
    )
    if cache_key not in heston_cache:
        calibration = _get_calibration(ticker)
        atm_iv = (
            None if expiry_date is None else YFProvider().get_atm_iv(ticker, expiry_date)
        )
        print(f"DEBUG: ATM IV for {ticker} @ {expiry_date}: {atm_iv}")
        # Three tiers in descending order of quality: a fit to the quoted surface,
        # the ATM implied variance with theta pinned to it, or realized volatility.
        # Only the last always works, which is what keeps the panel priced when the
        # market is closed or the ticker has no options at all.
        heston_params = estimate_heston_parameters(
            df_result,
            market_v0=None if atm_iv is None else atm_iv**2,
            tau=expiry_days / 365,
            calibration=calibration,
        )
        _cache_store(heston_cache, cache_key, (atm_iv, heston_params, calibration))
    return heston_cache[cache_key]


# CALLBACK: Price chart, forecast and the metrics derived from them.
# Split from the option panel below so that changing the expiry does not rebuild
# this figure, and changing the horizon or confidence level does not reach for the
# option chain over the network.
@app.callback(
    [
        Output("main-chart", "figure"),
        Output("metric-move", "children"),
        Output("metric-move-label", "children"),
        Output("metric-move-band", "children"),
        Output("metric-ticker", "children"),
        Output("metric-asof", "children"),
        Output("metric-asof", "style"),
        Output("metric-price", "children"),
        Output("metric-change", "children"),
        Output("metric-change", "style"),
        Output("metric-regime", "children"),
        Output("metric-regime-detail", "children"),
        Output("status-alert", "children", allow_duplicate=True),
        Output("status-alert", "color", allow_duplicate=True),
        Output("status-alert", "is_open", allow_duplicate=True),
    ],
    [
        Input("data-store", "data"),
        Input("vola-slider", "value"),
        Input("horizon-slider", "value"),
        Input("feature-toggles", "value"),
        Input("ci-level", "value"),
    ],
    [State("ticker-input", "value")],
    prevent_initial_call="initial_duplicate",
)
def update_price_panel(
    json_data, vola_val, horizon_val, selected_features, ci_level, ticker
):
    if json_data is None:
        raise exceptions.PreventUpdate
    try:
        return _build_price_panel(
            json_data, vola_val, horizon_val, selected_features, ci_level, ticker
        )
    except Exception as exc:
        # A too short date range (the indicators need ~200 rows) must not leave the
        # user staring at a stale chart with no explanation.
        print(f"ERROR: price panel update failed: {exc}")
        return (
            *(dash.no_update,) * 12,
            f"Could not update the charts: {exc}",
            "danger",
            True,
        )


def _build_price_panel(
    json_data, vola_val, horizon_val, selected_features, ci_level, ticker
):
    """
    Builds the price figure and its metrics.

    :param json_data: Serialized OHLCV DataFrame from the data store.
    :param vola_val: HMM volatility lookback window in trading days.
    :param horizon_val: Forecast horizon in trading days.
    :param selected_features: Optional features enabled in the UI.
    :param ci_level: Confidence level of the outer forecast band, in percent.
    :param ticker: Active ticker symbol.
    :return: Tuple matching the callback's output list.
    """
    df_result, data_id, summary = _get_regimes(json_data, vola_val)
    latest_regime_text = "Low Vol" if summary["current"] == 0 else "High Vol"

    forecaster = _get_forecaster(
        df_result, ticker, vola_val, selected_features, data_id
    )
    prediction_cum = forecaster.predict_latest(df_result)

    # Projection
    last_date = df_result.index[-1]
    last_price = float(df_result["Close"].iloc[-1])

    # The model predicts a cumulative log return over its own training horizon,
    # so scale it down to a per trading day drift before projecting it forward.
    daily_log_return = prediction_cum / forecaster.forecast_horizon

    # Trading days, not calendar days: the target is defined on the trading day
    # series, so calendar dates would stretch the projection across weekends.
    future_dates = pd.bdate_range(
        start=last_date + pd.offsets.BDay(1), periods=horizon_val
    )
    steps = np.arange(1, horizon_val + 1)
    future_prices = last_price * np.exp(daily_log_return * steps)

    # Lognormal confidence bands: the price is modelled as
    # last_price * exp(mu * i +- z * sigma * sqrt(i)), which keeps the bands
    # symmetric in log space and strictly positive.
    daily_vola = float(df_result["volatility"].iloc[-1])
    horizon_vola = daily_vola * np.sqrt(steps)
    bands = []
    for level in dict.fromkeys([ci_level, INNER_CI_LEVEL]):
        z_score = norm.ppf(0.5 + level / 200.0)
        bands.append(
            (
                float(level),
                future_prices * np.exp(-z_score * horizon_vola),
                future_prices * np.exp(z_score * horizon_vola),
            )
        )

    # Calculate Move for the selected horizon
    expected_pct_change = (np.exp(daily_log_return * horizon_val) - 1) * 100

    # The point estimate alone says nothing about how wide the outcome is, so report
    # the same outer band the chart draws, as a range around today's price.
    outer_level, outer_low, outer_high = max(bands, key=lambda band: band[0])
    band_text = (
        f"{(outer_low[-1] / last_price - 1) * 100:+.1f}% … "
        f"{(outer_high[-1] / last_price - 1) * 100:+.1f}% @ {outer_level:g}%"
    )

    day_change = (last_price / float(df_result["Close"].iloc[-2]) - 1) * 100
    change_style = {
        "fontSize": DETAIL_SIZE,
        "color": UP_COLOR if day_change >= 0 else DOWN_COLOR,
    }

    # How old the data is. The chart marks the last bar, so saying nothing here would
    # let a three day old series read as today.
    age_days = (pd.Timestamp.now().normalize() - last_date.normalize()).days
    stale = age_days > STALE_AFTER_DAYS
    asof_text = f"as of {last_date.date()}" + (f" · {age_days}d old" if stale else "")
    asof_style = {
        "fontSize": DETAIL_SIZE,
        "color": FALLBACK_COLOR if stale else LABEL_COLOR,
    }

    current = summary["current"]
    regime_detail = (
        f"{summary['run_length']}d in · typ. "
        f"{summary['expected_durations'][current]:.0f}d · "
        f"{summary['annualised_vols'][current]:.0%} vol"
    )

    fig_main = Visualizer.plot_price_forecast(
        df_result,
        ticker,
        future_dates,
        future_prices,
        bands,
        lookback_days=DEFAULT_LOOKBACK,
    )

    return (
        fig_main,
        f"{expected_pct_change:.2f}%",
        f"EXP. {horizon_val}D MOVE",
        band_text,
        ticker,
        asof_text,
        asof_style,
        f"{last_price:,.2f}",
        f"{day_change:+.2f}% d/d",
        change_style,
        latest_regime_text,
        regime_detail,
        "",
        "danger",
        False,
    )


# CALLBACK: Heston chart and the option metrics. Depends on the expiry, which the
# price panel above does not use, and is the only path that hits the network.
@app.callback(
    [
        Output("heston-chart", "figure"),
        Output("metric-call", "children"),
        Output("metric-put", "children"),
        Output("metric-iv", "children"),
        Output("label-call", "children"),
        Output("label-put", "children"),
        Output("label-iv", "children"),
        Output("metric-iv", "style"),
        Output("label-iv", "style"),
        Output("iv-tooltip", "children"),
        Output("option-alert", "children"),
        Output("option-alert", "color"),
        Output("option-alert", "is_open"),
    ],
    [
        Input("data-store", "data"),
        Input("vola-slider", "value"),
        Input("expiry-dropdown", "value"),
    ],
    [State("ticker-input", "value")],
)
def update_option_panel(json_data, vola_val, expiry_value, ticker):
    # None means load_expirations has not answered yet; NO_EXPIRY is a real answer.
    if json_data is None or expiry_value is None:
        raise exceptions.PreventUpdate
    try:
        return _build_option_panel(json_data, vola_val, expiry_value, ticker)
    except Exception as exc:
        print(f"ERROR: option panel update failed: {exc}")
        return (
            *(dash.no_update,) * 10,
            f"Could not price options: {exc}",
            "danger",
            True,
        )


def _build_option_panel(json_data, vola_val, expiry_value, ticker):
    """
    Builds the Heston price curve and the option metrics in the header.

    :param json_data: Serialized OHLCV DataFrame from the data store.
    :param vola_val: HMM volatility lookback window in trading days.
    :param expiry_value: A listed expiry date, or NO_EXPIRY for tickers with none.
    :param ticker: Active ticker symbol.
    :return: Tuple matching the callback's output list.
    """
    df_result, data_id, _ = _get_regimes(json_data, vola_val)

    # Without a listed expiry there is nothing to read an implied volatility from,
    # so price a default tenor off realized volatility rather than leave the panel
    # blank. max(1, ...) keeps tau positive if an expiry lapses while the page is open.
    expiry_date = None if expiry_value == NO_EXPIRY else expiry_value
    expiry_val = (
        DEFAULT_EXPIRY_DAYS
        if expiry_date is None
        else max(1, (pd.Timestamp(expiry_date) - pd.Timestamp.now().normalize()).days)
    )

    atm_iv, heston_params, calibration = _get_heston(
        df_result, ticker, expiry_date, expiry_val, vola_val, data_id
    )

    S0 = float(df_result["Close"].iloc[-1])
    strike_range = np.linspace(S0 * 0.8, S0 * 1.2, 50)
    strikes, call_prices = get_heston_fft_calls(
        **heston_params, interpolate_strikes=strike_range
    )
    _, put_prices = get_heston_fft_puts(
        **heston_params, interpolate_strikes=strike_range
    )
    # Quoted mids for the same contract the model just priced, so the panel shows
    # where the model sits against the market rather than only its own curve.
    if expiry_date is None:
        market = {}
    else:
        quotes_key = f"{ticker}_{expiry_date}_{_quote_bucket()}"
        if quotes_key not in quotes_cache:
            _cache_store(
                quotes_cache,
                quotes_key,
                YFProvider().get_quoted_prices(ticker, expiry_date),
            )
        market = quotes_cache[quotes_key]

    fig_heston = Visualizer.plot_heston_prices(
        strikes, call_prices, put_prices, S0, expiry_val, market=market
    )

    # Find ATM Call/Put price (index where strike is closest to S0)
    idx_atm = (np.abs(strikes - S0)).argmin()

    # The volatility the model actually priced this tenor with, whichever tier
    # supplied it, so the header never shows a number the model did not use.
    pricing_vola = effective_vola(heston_params)
    if calibration is not None:
        vola_label = f"CALIB. VOLA ({expiry_val}D)"
        vola_value_style = {"color": METRIC_COLOR}
        vola_label_style = {"color": LABEL_COLOR, "fontSize": CAPTION_SIZE}
        vola_tooltip = (
            f"Fitted to {ticker}'s quoted option surface: {calibration['n_quotes']} "
            f"out-of-the-money contracts across {calibration['n_maturities']} "
            f"expiries, with a price RMSE of {calibration['rmse_price']:.2f}. The fit "
            f"sets v0 to {np.sqrt(calibration['v0']):.1%} and the long-run level "
            f"theta to {np.sqrt(calibration['theta']):.1%}; the figure shown is the "
            f"variance the model expects over these {expiry_val} days, which is what "
            f"prices the curve below. Interest rate and dividend come from put-call "
            f"parity rather than an assumption."
        )
    elif atm_iv is None:
        # The chain gave us nothing usable, so these prices rest on realized
        # volatility. Amber marks the whole metric as a fallback.
        vola_label = f"REALIZED VOLA ({expiry_val}D)"
        vola_value_style = {"color": FALLBACK_COLOR}
        vola_label_style = {"color": FALLBACK_COLOR, "fontSize": CAPTION_SIZE}
        vola_tooltip = (
            f"No usable implied volatility for {ticker}: the option chain carries "
            f"no quotes - normal outside US trading hours - or Yahoo lists no "
            f"options for this ticker at all. The Heston prices and this "
            f"figure fall back to realized volatility, annualised from the last "
            f"{realized_vola_window(expiry_val / 365)} trading days of the loaded "
            f"history - the window matching the {expiry_val}-day tenor."
        )
    else:
        vola_label = f"IMPLIED VOLA ({expiry_val}D)"
        vola_value_style = {"color": METRIC_COLOR}
        vola_label_style = {"color": LABEL_COLOR, "fontSize": CAPTION_SIZE}
        vola_tooltip = (
            f"At-the-money implied volatility for {ticker}, averaged over the call "
            f"and the put nearest spot on the {expiry_date} chain. It sets v0 for "
            f"the Heston prices."
        )

    return (
        fig_heston,
        f"{call_prices[idx_atm]:.2f}",
        f"{put_prices[idx_atm]:.2f}",
        f"{pricing_vola:.2%}",
        f"FAIR ATM CALL ({expiry_val}D)",
        f"FAIR ATM PUT ({expiry_val}D)",
        vola_label,
        vola_value_style,
        vola_label_style,
        vola_tooltip,
        "",
        "danger",
        False,
    )


app.clientside_callback(
    Y_RESCALE_JS,
    Output("main-chart", "figure", allow_duplicate=True),
    Input("main-chart", "relayoutData"),
    State("main-chart", "figure"),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    debug_mode = os.environ.get("DASH_DEBUG", "false").lower() == "true"
    app.run(host="127.0.0.1", port=8050, debug=debug_mode)
