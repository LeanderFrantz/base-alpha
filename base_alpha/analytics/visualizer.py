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
    def plot_heston_prices(strikes, call_prices, put_prices, current_price, expiry_days):
        """
        Creates a line chart for Heston call and put prices with an ATM line.

        :param strikes: Array of strike prices.
        :param call_prices: Array of call prices.
        :param put_prices: Array of put prices.
        :param current_price: The current asset price (ATM).
        :param expiry_days: Days to expiry.
        :return: Plotly figure object.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strikes, y=call_prices, name="Call Prices", line=dict(color="#00d1b2")))
        fig.add_trace(go.Scatter(x=strikes, y=put_prices, name="Put Prices", line=dict(color="#ff5050")))
        
        # Add ATM dashed line
        fig.add_vline(x=current_price, line_dash="dash", line_color="white", annotation_text="ATM")
        
        fig.update_layout(
            title=f"Heston Fair Option Prices ({expiry_days}D)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1a1a2e",
            xaxis_title="Strike Price",
            yaxis_title="Option Price",
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig
