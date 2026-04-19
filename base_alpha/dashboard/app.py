import io
import dash
from dash import dcc, html, Input, Output, State, exceptions
import dash_bootstrap_components as dbc
from base_alpha.data_provider.data_provider import YFProvider
from base_alpha.models.regime_detection import RegimeDetector
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

app.layout = dbc.Container(
    [
        # Speicher für die Daten (JSON-Format)
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
                                            start_date="2018-01-01",
                                            end_date="2024-04-01",
                                            className="mb-3",
                                        ),
                                        dbc.Button(
                                            "Fetch New Data",
                                            id="fetch-btn",
                                            color="warning",
                                            className="w-100",
                                        ),
                                    ]
                                ),
                                dbc.CardHeader("Model Settings", className="mt-3"),
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
                    [dcc.Loading(dcc.Graph(id="main-chart", style={"height": "75vh"}))],
                    width=9,
                ),
            ]
        ),
    ],
    fluid=True,
)


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


# CALLBACK 2: Nur HMM rechnen (Triggert bei Slider-Änderung ODER neuen Daten)
@app.callback(
    Output("main-chart", "figure"),
    [Input("data-store", "data"), Input("vola-slider", "value")],
    [State("ticker-input", "value")],
)
def run_model_analysis(json_data, vola_val, ticker):
    if json_data is None:
        raise exceptions.PreventUpdate

    # Daten aus Store holen
    df_ohlcv = pd.read_json(io.StringIO(json_data), orient="split")

    print(f"DEBUG: Running HMM with Vola-Lookback: {vola_val}...")
    detector = RegimeDetector(n_regimes=2, vola_window=vola_val)
    df_result = detector.fit_predict(df_ohlcv)

    # Plotting (deine Subplot-Logik)
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
