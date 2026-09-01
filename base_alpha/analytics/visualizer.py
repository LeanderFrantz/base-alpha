import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Visualizer:
    """
    Provides interactive visualization tools for market data and model results.
    """

    @staticmethod
    def plot_regimes(df_result: pd.DataFrame, ticker: str):
        """
        Plots an interactive price chart with color-coded background for regimes.

        :param df_result: DataFrame containing 'Close' and 'Regime'.
        :param ticker: Ticker symbol for the title.
        """
        # create subplots for the probabilities of the regimes
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
        )

        # add clsoing price line
        fig.add_trace(
            go.Scatter(
                x=df_result.index,
                y=df_result["Close"],
                line=dict(color="black", width=1.5),
                name=ticker,
            ),
            row=1,
            col=1,
        )

        # add probability line for high volatility regime
        fig.add_trace(
            go.Scatter(
                x=df_result.index,
                y=df_result["Prob_HighVola"],
                fill="tozeroy",
                line=dict(color="red", width=1),
                name="Prob High Vola",
            ),
            row=2,
            col=1,
        )

        # add colored background for regimes (0 = low volatility, 1 = high volatility)
        df_result["change"] = df_result["Regime"].diff().fillna(0)
        change_indices = df_result.index[df_result["change"] != 0].tolist()
        change_indices = [df_result.index[0]] + change_indices + [df_result.index[-1]]

        for i in range(len(change_indices) - 1):
            start = change_indices[i]
            end = change_indices[i + 1]
            regime = df_result.loc[start, "Regime"]
            # green for low volatility, red for high volatility
            color = "rgba(0, 255, 0, 0.2)" if regime == 0 else "rgba(255, 0, 0, 0.2)"

            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=color,
                layer="below",
                line_width=0,
                row=1,
                col=1,
            )

        # update layout
        fig.update_layout(
            title=f"Hidden Market Regimes: {ticker}",
            yaxis_title="Price",
            template="plotly_white",
            hovermode="x unified",
            height=800,
            showlegend=False,
        )

        # update axis tiles for the subplots
        fig.update_yaxes(
            title_text="P(Regime=High Volatility)", range=[0, 1], row=2, col=1
        )
        fig.update_xaxes(title_text="Date", row=2, col=1)

        fig.show()

    @staticmethod
    def plot_heston_prices(
        strikes, call_prices, put_prices, current_price, expiry_days, market=None
    ):
        """
        Creates a line chart for Heston call and put prices with an ATM line.

        :param strikes: Array of strike prices.
        :param call_prices: Array of call prices.
        :param put_prices: Array of put prices.
        :param current_price: The current asset price (ATM).
        :param expiry_days: Days to expiry.
        :param market: Optional {"calls": (strikes, mids), "puts": (strikes, mids)}
            of quoted mid prices, drawn as markers so the gap to the model curve is
            visible. Omitted sides and an empty dict simply draw nothing.
        :return: Plotly figure object.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strikes, y=call_prices, name="Call Prices", line=dict(color="#00d1b2")))
        fig.add_trace(go.Scatter(x=strikes, y=put_prices, name="Put Prices", line=dict(color="#ff5050")))
        
        # Quoted mids on top of the model curve; the distance between them is the
        # whole point of the panel.
        for side, colour, label in (
            ("calls", "#00d1b2", "Market Call"),
            ("puts", "#ff5050", "Market Put"),
        ):
            if not market or side not in market:
                continue
            market_strikes, market_mids = market[side]
            inside = (market_strikes >= min(strikes)) & (market_strikes <= max(strikes))
            if not inside.any():
                continue
            fig.add_trace(
                go.Scatter(
                    x=market_strikes[inside],
                    y=market_mids[inside],
                    mode="markers",
                    name=label,
                    marker=dict(
                        color=colour, size=7, symbol="circle-open", line=dict(width=2)
                    ),
                )
            )

        # Add ATM dashed line
        fig.add_vline(x=current_price, line_dash="dash", line_color="white", annotation_text="ATM")

        title = f"Heston Fair Option Prices ({expiry_days}D)"
        if not market:
            title += " - no market quotes"

        fig.update_layout(
            title=title,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1a1a2e",
            xaxis_title="Strike Price",
            yaxis_title="Option Price",
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig

    @staticmethod
    def plot_price_forecast(
        df_result: pd.DataFrame,
        ticker: str,
        forecast_dates: pd.DatetimeIndex,
        forecast_prices: np.ndarray,
        bands: list[tuple[float, np.ndarray, np.ndarray]],
        lookback_days: int = 120,
    ) -> go.Figure:
        """
        Plots history, HMM regimes and the forward projection in a single figure.

        History and forecast share one x-axis, and the forecast traces are anchored
        on the last observed close so the projection continues the price line
        instead of starting next to it.

        :param df_result: DataFrame from RegimeDetector.fit_predict(), containing
            'Close', 'Regime' and 'Prob_HighVola'.
        :param ticker: Ticker symbol, used for the title and the price trace name.
        :param forecast_dates: Trading days the projection covers.
        :param forecast_prices: Projected median price for each forecast date.
        :param bands: One (confidence_level, lower_prices, upper_prices) tuple per
            band. Drawn widest first so narrower bands stay visible on top.
        :param lookback_days: Trading days of history shown in the default view.
        :return: Plotly figure object.
        """
        last_date = df_result.index[-1]
        last_price = float(df_result["Close"].iloc[-1])

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
        )

        # Historical close
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

        # Prepending the last observed point closes the gap between history and
        # forecast, and lets every band open out of a single point at "today".
        anchor_x = pd.DatetimeIndex([last_date]).append(
            pd.DatetimeIndex(forecast_dates)
        )

        def _anchored(values: np.ndarray) -> np.ndarray:
            return np.concatenate([[last_price], np.asarray(values, dtype=float)])

        # Confidence bands, widest first
        for idx, (level, lower, upper) in enumerate(
            sorted(bands, key=lambda band: band[0], reverse=True)
        ):
            fig.add_trace(
                go.Scatter(
                    x=anchor_x,
                    y=_anchored(lower),
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=anchor_x,
                    y=_anchored(upper),
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=f"rgba(0, 209, 178, {0.12 + 0.12 * idx:.2f})",
                    name=f"{level:g}% Confidence",
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

        # Median projection
        fig.add_trace(
            go.Scatter(
                x=anchor_x,
                y=_anchored(forecast_prices),
                mode="lines",
                name="Forecast",
                line=dict(color="#00d1b2", width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

        # High volatility probability
        fig.add_trace(
            go.Scatter(
                x=df_result.index,
                y=df_result["Prob_HighVola"],
                fill="tozeroy",
                name="P(High Vol)",
                line=dict(color="#ff5050", width=1),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Colored background per regime (0 = low volatility, 1 = high volatility)
        regimes = df_result["Regime"]
        change_points = (
            [df_result.index[0]]
            + regimes.index[regimes.diff().fillna(0) != 0].tolist()
            + [df_result.index[-1]]
        )
        for i in range(len(change_points) - 1):
            start, end = change_points[i], change_points[i + 1]
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=(
                    "rgba(0, 255, 100, 0.18)"
                    if regimes.loc[start] == 0
                    else "rgba(255, 80, 80, 0.45)"
                ),
                layer="below",
                line_width=0,
                row=1,
                col=1,
            )

        # Separate the projection from the realized history
        fig.add_vrect(
            x0=last_date,
            x1=anchor_x[-1],
            fillcolor="rgba(255, 255, 255, 0.05)",
            layer="below",
            line_width=0,
            row=1,
            col=1,
        )
        fig.add_vline(
            x=last_date,
            line_dash="dot",
            line_color="#6c757d",
            line_width=1,
            annotation_text=last_date.strftime("%Y-%m-%d"),
            annotation_position="top left",
            annotation_font=dict(color="#6c757d", size=10),
            row=1,
            col=1,
        )

        # Every view keeps the whole projection and varies only how much history
        # sits in front of it.
        x_end = anchor_x[-1] + pd.Timedelta(days=2)

        def _y_range(x_start: pd.Timestamp) -> list[float]:
            """Price range covering the history from x_start plus every band."""
            visible = df_result["Close"].loc[x_start:]
            low = min([visible.min()] + [float(np.min(b)) for _, b, _ in bands])
            high = max([visible.max()] + [float(np.max(b)) for _, _, b in bands])
            pad = (high - low) * 0.08
            return [low - pad, high + pad]

        # Default view: recent history plus the projection, so the bands stay readable.
        x_range = [df_result["Close"].tail(lookback_days).index[0], x_end]

        fig.update_layout(
            # The view buttons and the legend share a strip just above the plot
            # area, so the title is pinned to the top of the container instead of
            # being centred in the top margin, where it collided with them.
            title=dict(
                text=f"{ticker} | Price, HMM Regime & Forecast",
                x=0,
                xanchor="left",
                y=0.97,
                yanchor="top",
                yref="container",
            ),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1a1a2e",
            hovermode="x unified",
            # plotly_dark leaves hoverlabel.bgcolor unset, and the unified hover box
            # otherwise derives its background from paper_bgcolor - which is
            # transparent here, leaving light text on a near-white box.
            hoverlabel=dict(
                bgcolor="#1a1a2e",
                bordercolor="#00d1b2",
                font=dict(color="#E0E0E0", size=12),
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(l=40, r=40, t=120, b=40),
        )
        # Plotly's own range selector always counts back from the right edge of the
        # axis, which is the end of the forecast - "1M" would then buy its month
        # partly from the projection and leave barely three weeks of history. These
        # buttons count back from the last observed day instead and keep the whole
        # projection on top, and each one carries the matching price range so the
        # chart does not stay scaled to the previous window.
        view_windows = [
            ("1M", pd.DateOffset(months=1)),
            ("3M", pd.DateOffset(months=3)),
            ("6M", pd.DateOffset(months=6)),
            ("1Y", pd.DateOffset(years=1)),
            ("All", None),
        ]
        buttons = []
        for label, offset in view_windows:
            x_start = (
                df_result.index[0]
                if offset is None
                else max(last_date - offset, df_result.index[0])
            )
            buttons.append(
                dict(
                    label=label,
                    method="relayout",
                    args=[
                        {
                            "xaxis.range": [x_start.isoformat(), x_end.isoformat()],
                            "xaxis2.range": [x_start.isoformat(), x_end.isoformat()],
                            "yaxis.range": _y_range(x_start),
                        }
                    ],
                )
            )

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    buttons=buttons,
                    # No button matches the default lookback exactly, so start with
                    # none of them highlighted.
                    active=-1,
                    showactive=True,
                    bgcolor="#1a1a2e",
                    bordercolor="#00d1b2",
                    borderwidth=1,
                    font=dict(color="#E0E0E0", size=10),
                    pad=dict(l=2, r=2, t=2, b=2),
                    x=0,
                    xanchor="left",
                    y=1.02,
                    yanchor="bottom",
                )
            ]
        )
        fig.update_xaxes(range=x_range, row=1, col=1)
        fig.update_xaxes(range=x_range, row=2, col=1)
        fig.update_yaxes(title="Price", range=_y_range(x_range[0]), row=1, col=1)
        fig.update_yaxes(title="P(High Vol)", range=[0, 1], row=2, col=1)

        return fig
