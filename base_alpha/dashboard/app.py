import io
import dash
from dash import dcc, html, Input, Output, State, exceptions
import dash_bootstrap_components as dbc
from base_alpha.data_provider.data_provider import YFProvider
from base_alpha.models.regime_detection import RegimeDetector
from base_alpha.models.forecaster import Forecaster
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# Default dates: Today - 5 years to Today
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

app.layout = dbc.Container(
    [
        # Speicher für die Daten (JSON-Format)
        dcc.Store(id="data-store"),
        dcc.Store(id="forecaster-store"),
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
                                            value="^GSPC",
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
                                        html.Label("Forecast Horizon (Days):", className="mt-3"),
                                        dcc.Slider(
                                            id="horizon-slider",
                                            min=1,
                                            max=30,
                                            step=1,
                                            value=5,
                                            marks={i: str(i) for i in range(5, 31, 5)},
                                        ),
                                    ]
                                ),
                                dbc.CardHeader("Predictive Feature Selection"),
                                dbc.CardBody(
                                    [
                                        dbc.Checklist(
                                            id="feature-toggle-all",
                                            options=[{"label": "Enable/Disable All", "value": "all"}],
                                            value=["all"],
                                            inline=True,
                                            className="mb-2",
                                        ),
                                        dbc.Checklist(
                                            id="feature-toggles",
                                            options=[
                                                {"label": "RSI_14", "value": "RSI_14"},
                                                {"label": "dist_sma_200", "value": "dist_sma_200"},
                                                {"label": "MACD_12_26_9", "value": "MACD_12_26_9"},
                                                {"label": "LSTM_Feature", "value": "LSTM_Feature"},
                                            ],
                                            value=["RSI_14", "dist_sma_200", "MACD_12_26_9", "LSTM_Feature"],
                                            className="mb-2",
                                        ),
                                    ]
                                ),
                                dbc.CardHeader("HMM Settings", className="mt-3"),
                                dbc.CardBody(
                                    [
                                        html.Label("Volatility Lookback (Days):"),
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
                                        dbc.Col(html.Div([html.Small("ACTIVE TICKER", style={"color": "#6c757d", "fontSize": "0.75rem"}), html.H5(id="metric-ticker", children="-", style={"color": "#00d1b2"})]), width=2),
                                        dbc.Col(html.Div([html.Small(id="metric-move-label", children="EXP. MOVE", style={"color": "#6c757d", "fontSize": "0.75rem"}), html.H5(id="metric-move", children="-", style={"color": "#00d1b2"})]), width=2),
                                        dbc.Col(html.Div([html.Small("CURRENT REGIME", style={"color": "#6c757d", "fontSize": "0.75rem"}), html.H5(id="metric-regime", children="-", style={"color": "#00d1b2"})]), width=2),
                                        dbc.Col(html.Div([html.Small("FAIR CALL (30D)", style={"color": "#6c757d", "fontSize": "0.75rem"}), html.H5(id="metric-call", children="-", style={"color": "#00d1b2"})]), width=3),
                                        dbc.Col(html.Div([html.Small("FAIR PUT (30D)", style={"color": "#6c757d", "fontSize": "0.75rem"}), html.H5(id="metric-put", children="-", style={"color": "#00d1b2"})]), width=3),
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
                        dcc.Loading(dcc.Graph(id="main-chart", style={"height": "75vh"})),
                    ],
                    width=9,
                ),
            ]
        ),
    ],
    fluid=True,
)


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


# CALLBACK 1: Nur Daten laden
@app.callback(
    Output("data-store", "data"),
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
    df = provider.fetch_data(ticker, start, end)
    # DataFrame als JSON für den Store umwandeln
    return df.to_json(date_format="iso", orient="split")


# CALLBACK: Train Forecaster (Cached)
@app.callback(
    Output("forecaster-store", "data"),
    [Input("data-store", "data"), Input("feature-toggles", "value")],
)
def train_forecaster_callback(json_data, selected_features):
    if json_data is None:
        raise exceptions.PreventUpdate
    
    # Simple training flow (re-trains on feature toggles)
    # In a real app, you might want to cache this more intelligently
    df_ohlcv = pd.read_json(io.StringIO(json_data), orient="split")
    # For now, train on full data without regime window to avoid dependency on regime slider
    # (or you need to feed regime detector output here as well)
    
    # For demo: instantiate and train
    # Note: Training is still expensive. This makes feature toggling slow but chart updates fast.
    return "trained" # Dummy for now, we need a better way to serialize/cache the model
# Global cache for the forecaster model
model_cache = {}

# CALLBACK: Chart Analysis (Slow/Heavy)
@app.callback(
    Output("main-chart", "figure"),
    [
        Input("data-store", "data"), 
        Input("vola-slider", "value")
    ],
    [State("ticker-input", "value")],
)
def update_chart(json_data, vola_val, ticker):
    if json_data is None:
        raise exceptions.PreventUpdate

    df_ohlcv = pd.read_json(io.StringIO(json_data), orient="split")

    # 1. Regime Detection (Necessary for chart)
    detector = RegimeDetector(n_regimes=2, vola_window=vola_val)
    df_result = detector.fit_predict(df_ohlcv)

    # Plotting
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3]
    )

    fig.add_trace(
        go.Scatter(
            x=df_result.index,
            y=df_result["Close"],
            name=ticker,
            line=dict(color="#E0E0E0", width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_result.index,
            y=df_result["Prob_HighVola"],
            fill="tozeroy",
            name="Prob",
            line=dict(color="red"),
        ),
        row=2,
        col=1,
    )

    # Hintergründe
    df_result["change"] = df_result["Regime"].diff().fillna(0)
    pts = (
        [df_result.index[0]]
        + df_result.index[df_result["change"] != 0].tolist()
        + [df_result.index[-1]]
    )
    for i in range(len(pts) - 1):
        s, e = pts[i], pts[i + 1]
        color = (
            "rgba(0, 255, 100, 0.18)"
            if df_result.loc[s, "Regime"] == 0
            else "rgba(255, 80, 80, 0.45)"
        )
        fig.add_vrect(
            x0=s, x1=e, fillcolor=color, layer="below", line_width=0, row=1, col=1
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1a1a2e",
        showlegend=False,
    )
    return fig

# CALLBACK: Metrics Bar (Fast)
@app.callback(
    [
        Output("metric-move", "children"),
        Output("metric-move-label", "children"),
        Output("metric-ticker", "children"),
        Output("metric-regime", "children"),
    ],
    [
        Input("data-store", "data"), 
        Input("vola-slider", "value"),
        Input("horizon-slider", "value"),
        Input("feature-toggles", "value")
    ],
    [State("ticker-input", "value")],
)
def update_metrics(json_data, vola_val, horizon_val, selected_features, ticker):
    if json_data is None:
        raise exceptions.PreventUpdate

    df_ohlcv = pd.read_json(io.StringIO(json_data), orient="split")
    
    # 1. Regime Detection (Needed for Regime text & Forecaster)
    detector = RegimeDetector(n_regimes=2, vola_window=vola_val)
    df_result = detector.fit_predict(df_ohlcv)
    latest_regime_val = int(df_result["Regime"].iloc[-1])
    latest_regime_text = "Low Vol" if latest_regime_val == 0 else "High Vol"

    # 2. Forecaster Prediction (Cached)
    cache_key = f"{ticker}_{vola_val}_{str(selected_features)}"
    if cache_key not in model_cache:
        forecaster = Forecaster(forecast_horizon=5)
        mandatory_features = ["Regime", "Prob_LowVola", "Prob_HighVola"]
        active_features = mandatory_features + selected_features
        forecaster.feature_cols_xgb = active_features
        forecaster.train(df_result)
        model_cache[cache_key] = forecaster
    else:
        forecaster = model_cache[cache_key]
    
    prediction_1d = forecaster.predict_latest(df_result)
    prediction_scaled = prediction_1d * (horizon_val / 5.0) 
    
    expected_pct_change = (np.exp(prediction_scaled) - 1) * 100
    metric_move_text = f"{expected_pct_change:.2f}%"
    metric_move_label = f"EXP. {horizon_val}D MOVE"

    return metric_move_text, metric_move_label, ticker, latest_regime_text



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
