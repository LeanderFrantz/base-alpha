import io
import os
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

# Upper bound for the module level caches below.
MAX_CACHE_ENTRIES = 8

# Header colours. Amber flags a metric that is not a market quote but a fallback,
# so a substituted number can never pass for a real one.
METRIC_COLOR = "#00d1b2"
LABEL_COLOR = "#6c757d"
FALLBACK_COLOR = "#f0ad4e"

# Shortest history the feature pipeline can produce a complete row from.
MIN_HISTORY = SMA_WINDOW + TRAIN_HORIZON

# Default dates: Today - 5 years to Today
end_date = datetime.now() + timedelta(
    days=1
)  # Adjusted to ensure we have data for today
start_date = end_date - timedelta(days=5 * 365)

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
                                        html.Label("Option Expiry (Days):"),
                                        dcc.Slider(
                                            id="expiry-slider",
                                            min=1,
                                            max=365,
                                            step=1,
                                            value=30,
                                            marks={
                                                i: str(i) for i in range(30, 361, 60)
                                            },
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
                                        dbc.Col(
                                            html.Div(
                                                [
                                                    html.Small(
                                                        "ACTIVE TICKER",
                                                        style={
                                                            "color": "#6c757d",
                                                            "fontSize": "0.75rem",
                                                        },
                                                    ),
                                                    html.H5(
                                                        id="metric-ticker",
                                                        children="-",
                                                        style={"color": "#00d1b2"},
                                                    ),
                                                ]
                                            ),
                                            width=2,
                                        ),
                                        dbc.Col(
                                            html.Div(
                                                [
                                                    html.Small(
                                                        id="metric-move-label",
                                                        children="EXP. MOVE",
                                                        style={
                                                            "color": "#6c757d",
                                                            "fontSize": "0.75rem",
                                                        },
                                                    ),
                                                    html.H5(
                                                        id="metric-move",
                                                        children="-",
                                                        style={"color": "#00d1b2"},
                                                    ),
                                                ]
                                            ),
                                            width=2,
                                        ),
                                        dbc.Col(
                                            html.Div(
                                                [
                                                    html.Small(
                                                        "CURRENT REGIME",
                                                        style={
                                                            "color": "#6c757d",
                                                            "fontSize": "0.75rem",
                                                        },
                                                    ),
                                                    html.H5(
                                                        id="metric-regime",
                                                        children="-",
                                                        style={"color": "#00d1b2"},
                                                    ),
                                                ]
                                            ),
                                            width=2,
                                        ),
                                        dbc.Col(
                                            html.Div(
                                                [
                                                    html.Small(
                                                        "FAIR CALL (30D)",
                                                        id="label-call",
                                                        style={
                                                            "color": "#6c757d",
                                                            "fontSize": "0.75rem",
                                                        },
                                                    ),
                                                    html.H5(
                                                        id="metric-call",
                                                        children="-",
                                                        style={"color": "#00d1b2"},
                                                    ),
                                                ]
                                            ),
                                            width=2,
                                        ),
                                        dbc.Col(
                                            html.Div(
                                                [
                                                    html.Small(
                                                        "FAIR PUT (30D)",
                                                        id="label-put",
                                                        style={
                                                            "color": "#6c757d",
                                                            "fontSize": "0.75rem",
                                                        },
                                                    ),
                                                    html.H5(
                                                        id="metric-put",
                                                        children="-",
                                                        style={"color": "#00d1b2"},
                                                    ),
                                                ]
                                            ),
                                            width=2,
                                        ),
                                        dbc.Col(
                                            html.Div(
                                                [
                                                    html.Small(
                                                        "IMPLIED VOLA (30D)",
                                                        id="label-iv",
                                                        style={
                                                            "color": "#6c757d",
                                                            "fontSize": "0.75rem",
                                                        },
                                                    ),
                                                    html.H5(
                                                        id="metric-iv",
                                                        children="-",
                                                        style={"color": "#00d1b2"},
                                                    ),
                                                ]
                                            ),
                                            width=2,
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
    Input("fetch-btn", "n_clicks"),
    [
        State("ticker-input", "value"),
        State("date-picker", "start_date"),
        State("date-picker", "end_date"),
    ],
)
def fetch_data(n_clicks, ticker, start, end):

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


# Global cache for the forecaster model
model_cache = {}

# Global cache for Heston parameters
heston_cache = {}


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


# CALLBACK: Main Chart, Heston Chart & Metrics
@app.callback(
    [
        Output("main-chart", "figure"),
        Output("heston-chart", "figure"),
        Output("metric-move", "children"),
        Output("metric-move-label", "children"),
        Output("metric-ticker", "children"),
        Output("metric-regime", "children"),
        Output("metric-call", "children"),
        Output("metric-put", "children"),
        Output("metric-iv", "children"),
        Output("label-call", "children"),
        Output("label-put", "children"),
        Output("label-iv", "children"),
        Output("metric-iv", "style"),
        Output("label-iv", "style"),
        Output("iv-tooltip", "children"),
        Output("status-alert", "children", allow_duplicate=True),
        Output("status-alert", "color", allow_duplicate=True),
        Output("status-alert", "is_open", allow_duplicate=True),
    ],
    [
        Input("data-store", "data"),
        Input("vola-slider", "value"),
        Input("horizon-slider", "value"),
        Input("expiry-slider", "value"),
        Input("feature-toggles", "value"),
        Input("ci-level", "value"),
    ],
    [State("ticker-input", "value")],
    prevent_initial_call="initial_duplicate",
)
def update_all(
    json_data, vola_val, horizon_val, expiry_val, selected_features, ci_level, ticker
):
    if json_data is None:
        raise exceptions.PreventUpdate

    try:
        return _build_dashboard(
            json_data,
            vola_val,
            horizon_val,
            expiry_val,
            selected_features,
            ci_level,
            ticker,
        )
    except Exception as exc:
        # A too short date range (the indicators need ~200 rows) or a ticker
        # without an option chain must not leave the user staring at a stale
        # chart with no explanation.
        print(f"ERROR: dashboard update failed: {exc}")
        return (
            *(dash.no_update,) * 15,
            f"Could not update the dashboard: {exc}",
            "danger",
            True,
        )


def _build_dashboard(
    json_data, vola_val, horizon_val, expiry_val, selected_features, ci_level, ticker
):
    """
    Runs the full pipeline and assembles every output of the update_all callback.

    :param json_data: Serialized OHLCV DataFrame from the data store.
    :param vola_val: HMM volatility lookback window in trading days.
    :param horizon_val: Forecast horizon in trading days.
    :param expiry_val: Option expiry in calendar days.
    :param selected_features: Optional features enabled in the UI.
    :param ci_level: Confidence level of the outer forecast band, in percent.
    :param ticker: Active ticker symbol.
    :return: Tuple matching the callback's output list.
    """
    df_ohlcv = pd.read_json(io.StringIO(json_data), orient="split")
    if len(df_ohlcv) < MIN_HISTORY:
        raise ValueError(
            f"only {len(df_ohlcv)} trading days loaded, but the {SMA_WINDOW}-day SMA "
            f"feature needs at least {MIN_HISTORY}. Widen the date range and fetch again."
        )

    # 1. Regime Detection
    detector = RegimeDetector(n_regimes=2, vola_window=vola_val)
    df_result = detector.fit_predict(df_ohlcv)
    latest_regime_val = int(df_result["Regime"].iloc[-1])
    latest_regime_text = "Low Vol" if latest_regime_val == 0 else "High Vol"

    # Create a stable identifier for the dataset
    data_id = f"{len(df_ohlcv)}_{df_ohlcv.index[0]}_{df_ohlcv.index[-1]}"

    # 2. Heston Parameter Estimation
    # The parameters are derived from df_result, so the key has to cover every
    # input that changes it - ticker and expiry alone would serve stale values
    # after the volatility slider or the date range moved.
    heston_key = f"{ticker}_{expiry_val}_{vola_val}_{data_id}"
    if heston_key not in heston_cache:
        provider = YFProvider()
        atm_iv = provider.get_atm_iv(ticker, expiry_val)
        print(f"DEBUG: Fetched ATM IV for {ticker}: {atm_iv}")
        # A None here means the chain gave us nothing usable; estimate_heston_parameters
        # then derives v0 from realized volatility instead of an invented constant.
        heston_params = estimate_heston_parameters(
            df_result,
            market_v0=None if atm_iv is None else atm_iv**2,
            tau=expiry_val / 365,
        )
        _cache_store(heston_cache, heston_key, (atm_iv, heston_params))
    else:
        atm_iv, heston_params = heston_cache[heston_key]
        print(f"DEBUG: Using cached Heston parameters for {heston_key}")

    # 3. Forecaster Prediction
    cache_key = f"{ticker}_{vola_val}_{str(selected_features)}_{data_id}"

    if cache_key not in model_cache:
        forecaster = Forecaster(forecast_horizon=TRAIN_HORIZON)
        active_features = MANDATORY_FEATURES + list(selected_features or [])
        forecaster.feature_cols_xgb = active_features
        forecaster.train(df_result)
        _cache_store(model_cache, cache_key, forecaster)
    else:
        forecaster = model_cache[cache_key]

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
    prediction_for_horizon = daily_log_return * horizon_val
    expected_pct_change = (np.exp(prediction_for_horizon) - 1) * 100
    metric_move_text = f"{expected_pct_change:.2f}%"
    metric_move_label = f"EXP. {horizon_val}D MOVE"

    # Main Chart: history, regimes and the projection in one figure
    fig_main = Visualizer.plot_price_forecast(
        df_result,
        ticker,
        future_dates,
        future_prices,
        bands,
        lookback_days=DEFAULT_LOOKBACK,
    )

    # Heston Chart
    S0 = last_price
    strike_range = np.linspace(S0 * 0.8, S0 * 1.2, 50)

    strikes, call_prices = get_heston_fft_calls(
        **heston_params, interpolate_strikes=strike_range
    )
    _, put_prices = get_heston_fft_puts(
        **heston_params, interpolate_strikes=strike_range
    )

    fig_heston = Visualizer.plot_heston_prices(
        strikes, call_prices, put_prices, S0, expiry_val
    )

    # Metrics
    # Find ATM Call/Put price (index where strike is closest to S0)
    idx_atm = (np.abs(strikes - S0)).argmin()
    metric_call_text = f"{call_prices[idx_atm]:.2f}"
    metric_put_text = f"{put_prices[idx_atm]:.2f}"

    # heston_params["v0"] is by construction the variance these prices came from,
    # whichever source supplied it, so the header never shows a number the model
    # did not use.
    pricing_vola = float(np.sqrt(heston_params["v0"]))
    if atm_iv is None:
        # The chain gave us nothing usable, so these prices rest on realized
        # volatility. Amber marks the whole metric as a fallback.
        vola_label = f"REALIZED VOLA ({expiry_val}D)"
        vola_value_style = {"color": FALLBACK_COLOR}
        vola_label_style = {"color": FALLBACK_COLOR, "fontSize": "0.75rem"}
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
        vola_label_style = {"color": LABEL_COLOR, "fontSize": "0.75rem"}
        vola_tooltip = (
            f"At-the-money implied volatility for {ticker}, averaged over the call "
            f"and the put nearest spot on the listed expiry closest to "
            f"{expiry_val} days. It sets v0 for the Heston prices."
        )

    return (
        fig_main,
        fig_heston,
        metric_move_text,
        metric_move_label,
        ticker,
        latest_regime_text,
        metric_call_text,
        metric_put_text,
        f"{pricing_vola:.2%}",
        f"FAIR CALL ({expiry_val}D)",
        f"FAIR PUT ({expiry_val}D)",
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
