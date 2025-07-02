import numpy as np
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
from portfolio import Portfolio
from black_scholes import calculate_call_data, calculate_put_data

# Constants
DEFAULT_FORWARD_PRICE = 100
DEFAULT_STRIKE_PRICE = 100
DEFAULT_TIME_TO_MATURITY = 1
DEFAULT_RISK_FREE_RATE = 0.05
DEFAULT_VOLATILITY = 0.35
DEFAULT_VOL_DELTA = 5
DEFAULT_MARKET_DELTA = 5
DEFAULT_QUANTITY = 1000
HEATMAP_HEIGHT = '80vh'



scenarios = {}


def validate_inputs(F, K, T, r, sigma):
    """Validate Black-Scholes inputs"""
    if None in [F, K, T, r, sigma]:
        return False #, "All parameters must be provided"
    if T <= 0:
        return False #, "Time to maturity must be positive"
    if sigma <= 0:
        return False #, "Volatility must be positive"
    return True #, ""


def calculate_scenario(F, K, T, r, sigma, option_type, vol_delta=0, market_delta=0, quantity=1):
    """
    Calculate the scenario based on the given parameters.
    """
    if option_type == 'call':
        price, delta, gamma, theta, vega = calculate_call_data(F, K, T, r, sigma, vol_delta=vol_delta, market_delta=market_delta)
    else:
        price, delta, gamma, theta, vega = calculate_put_data(F, K, T, r, sigma, vol_delta=vol_delta, market_delta=market_delta)
    premium = price * quantity
    delta = delta * quantity
    gamma = gamma * quantity
    theta = theta * quantity
    vega = vega * quantity
    portfolio = Portfolio(price, delta, gamma, theta, vega, premium)
    portfolio.calculate_p_and_l(price * quantity)
    return portfolio


app = Dash()

app.layout = [
    html.H1(children='Welcome to the Black-Scholes Tool'),
    html.Div(children='This tool allows you to calculate Black-Scholes option pricing.'),

    html.Label('Adjust Volatility'),
    dcc.Input(
        id='vol_delta',
        type='number',
        value=DEFAULT_VOL_DELTA,
        min=0,
        #max=100,
        step=1,
        style={'margin': '20px'}
    ),

    html.Label('Adjust Market'),
    dcc.Input(
        id='market_delta',
        type='number',
        value=DEFAULT_MARKET_DELTA,
        min=0,
        #max=100,
        step=1,
        style={'margin': '20px'}
    ),
    html.Label("Contracts"),
    dcc.Input(
        id='quantity',
        type='number',
        value=1000,
        min=1,
        step=1),

    html.Button('Calculate', id='calculate-button', style={'margin': '10px', 'padding': '10px 20px',
                                                           'backgroundColor': '#4CAF50'}),
    dcc.RadioItems(options=['put', 'call'], value='call', id='option-type'),

    html.Hr(),

    html.Div(children='Input Parameters for Black-Scholes Calculation:'),
    html.Div([
        html.Label('F (Forward Price)', style={'margin': '5px'}),
        dcc.Input(id='input-F',
                  type='number',
                  value=DEFAULT_FORWARD_PRICE,
                  min=0,
                  step=1,
                  placeholder='F (Forward Price)',
                  style={'margin': '5px'}),

        html.Label('K (Strike Price)', style={'margin': '5px'}),
        dcc.Input(id='input-K',
                  type='number',
                  value=DEFAULT_STRIKE_PRICE,
                  min=0,
                  step=1,
                  placeholder='K (Strike Price)',
                  style={'margin': '5px'}),

        html.Label('T (Time to Maturity)', style={'margin': '5px'}),
        dcc.Input(id='input-T',
                  type='number',
                  value=DEFAULT_TIME_TO_MATURITY,
                  min=0,
                  step=0.01,
                  placeholder='T (Time to Maturity)',
                  style={'margin': '5px'}),

        html.Label('r (Risk-Free Rate)',
                   style={'margin': '5px'}),
        dcc.Input(id='input-r',
                  type='number',
                  value=DEFAULT_RISK_FREE_RATE,
                  min=0,
                  step=0.01,
                  placeholder='r (Risk-Free Rate)',
                  style={'margin': '5px'}),

        html.Label('σ (Volatility)',
                   style={'margin': '5px'}),

        dcc.Input(id='input-sigma',
                  type='number',
                  value=DEFAULT_VOLATILITY,
                  min=0,
                  step=0.01,
                  placeholder='σ (Volatility)',
                  style={'margin': '5px'})

    ], style={'display': 'flex', 'flexDirection': 'column'}),  # Flexbox for horizontal layout
    html.Hr(),

    dcc.Graph(
        id='heatmap',
        figure={},
        style={'height': '80vh', 'width': '100%'}
    )
]


@callback(
    Output('heatmap', 'figure'),
    Input('input-F', 'value'),
    Input('input-K', 'value'),
    Input('input-T', 'value'),
    Input('input-r', 'value'),
    Input('input-sigma', 'value'),
    Input('option-type', 'value'),
    Input('calculate-button', 'n_clicks'),
    Input('vol_delta', 'value'),
    Input('market_delta', 'value'),
    Input('quantity', 'value')
)
def update_graph(F, K, T, r, sigma, option_type, n_clicks, vol_delta, market_delta, quantity):
#def update_graph(*args):

    try:
        #F, K, T, r, sigma, option_type, n_clicks, vol_delta, market_delta, quantity = args
        # dev test
        if n_clicks is None or n_clicks == 0:
            return go.Figure()  # Return an empty figure if no clicks

        # is_valid = validate_inputs(F, K, T, r, sigma)
        # if not is_valid:
        #     return go.Figure()  # Return an empty figure if inputs are invalid

        # if n_clicks >= 1 and (F is None and K is None and T is None and r is None and sigma is None):
        #     F = DEFAULT_FORWARD_PRICE
        #     K = DEFAULT_STRIKE_PRICE
        #     T = DEFAULT_TIME_TO_MATURITY
        #     r = DEFAULT_RISK_FREE_RATE
        #     sigma = DEFAULT_VOLATILITY
        #     vol_delta = DEFAULT_VOL_DELTA
        #     market_delta = DEFAULT_MARKET_DELTA
        #     quantity = DEFAULT_QUANTITY

        original_portfolio = calculate_scenario(F, K, T, r, sigma, option_type, 0, 0, quantity)
        original_premium = original_portfolio.price * quantity



        # Calculate scenarios
        for vol_change in (-vol_delta, 0, vol_delta):
            for market_change in (-market_delta, 0, market_delta):
                portfolio = calculate_scenario(F, K, T, r, sigma, option_type, vol_change, market_change, quantity)
                portfolio.calculate_p_and_l(original_premium)
                scenarios[(vol_change, market_change)] = portfolio

        vol_levels = [-vol_delta, 0, vol_delta]
        market_levels = [-market_delta, 0, market_delta]

        cell_text = []
        p_and_l_data = []

        # Create cell text for the heatmap using Portfolio objects
        for vol_change in vol_levels:
            row_text = []
            row_data = []
            for market_change in market_levels:
                portfolio = scenarios[(vol_change, market_change)]
                row_text.append(portfolio.to_plotly_format())
                row_data.append(portfolio.p_and_l)
            cell_text.append(row_text)
            p_and_l_data.append(row_data)

        # Create a grid of data for the heatmap
        # p_and_l_data = np.array([
        #     [scenarios[(-vol_delta, -market_delta)], scenarios[(-vol_delta, 0)], scenarios[(-vol_delta, market_delta)]],
        #     [scenarios[(0, -market_delta)], scenarios[(0, 0)], scenarios[(0, market_delta)]],
        #     [scenarios[(vol_delta, -market_delta)], scenarios[(vol_delta, 0)], scenarios[(vol_delta, market_delta)]]
        # ])
        # cell_text = np.array([
        #     [scenarios[(-vol_delta, -market_delta)], scenarios[(-vol_delta, 0)], scenarios[(-vol_delta, market_delta)]],
        #     [scenarios[(0, -market_delta)], scenarios[(0, 0)], scenarios[(0, market_delta)]],
        #     [scenarios[(vol_delta, -market_delta)], scenarios[(vol_delta, 0)], scenarios[(vol_delta, market_delta)]]
        # ])
        #
        #
        #
        #
        #
        # # Calculate same_same (center - no changes)
        # # same_same = Portfolio(original_premium, delta, gamma, theta, vega, quantity)
        # # same_same.calculate_p_and_l(original_premium)
        #
        # # Market up, vol same (center right)
        # price, delta, gamma, theta, vega = calculate_call_data(F, K, T, r, sigma,
        #                                                        market_delta=market_delta) if option_type == 'call' else calculate_put_data(
        #     F, K, T, r, sigma, market_delta=market_delta)
        # premium = price * quantity
        # up_same = Portfolio(premium, delta, gamma, theta, vega, quantity)
        # up_same.calculate_p_and_l(original_premium)
        #
        #
        # # Market down, vol same (center left)
        # price, delta, gamma, theta, vega = calculate_call_data(F, K, T, r, sigma,
        #                                                        market_delta=-market_delta) if option_type == 'call' else calculate_put_data(
        #     F, K, T, r, sigma, market_delta=-market_delta)
        # premium = price * quantity
        # down_same = Portfolio(premium, delta, gamma, theta, vega, quantity)
        # down_same.calculate_p_and_l(original_premium)
        #
        #
        # # Market up, vol up (top right)
        # price, delta, gamma, theta, vega = calculate_call_data(F, K, T, r, sigma, vol_delta=vol_delta,
        #                                                        market_delta=market_delta) if option_type == 'call' else calculate_put_data(
        #     F, K, T, r, sigma, vol_delta=vol_delta, market_delta=market_delta)
        # premium = price * quantity
        # up_up = Portfolio(premium, delta, gamma, theta, vega, quantity)
        # up_up.calculate_p_and_l(original_premium)
        #
        #
        # # Market down, vol down (bottom left)
        # price, delta, gamma, theta, vega = calculate_call_data(F, K, T, r, sigma, vol_delta=-vol_delta,
        #                                                        market_delta=-market_delta) if option_type == 'call' else calculate_put_data(
        #     F, K, T, r, sigma, vol_delta=-vol_delta, market_delta=-market_delta)
        # premium = price * quantity
        # down_down = Portfolio(premium, delta, gamma, theta, vega, quantity)
        # down_down.calculate_p_and_l(original_premium)
        #
        #
        # # Market down, vol up (top left)
        # price, delta, gamma, theta, vega = calculate_call_data(F, K, T, r, sigma, vol_delta=vol_delta,
        #                                                        market_delta=-market_delta) if option_type == 'call' else calculate_put_data(
        #     F, K, T, r, sigma, vol_delta=vol_delta, market_delta=-market_delta)
        # premium = price * quantity
        # down_up = Portfolio(premium, delta, gamma, theta, vega, quantity)
        # down_up.calculate_p_and_l(original_premium)
        #
        #
        # # Market up, vol down (bottom right)
        # price, delta, gamma, theta, vega = calculate_call_data(F, K, T, r, sigma, vol_delta=-vol_delta,
        #                                                        market_delta=market_delta) if option_type == 'call' else calculate_put_data(
        #     F, K, T, r, sigma, vol_delta=-vol_delta, market_delta=market_delta)
        # premium = price * quantity
        # up_down = Portfolio(premium, delta, gamma, theta, vega, quantity)
        # up_down.calculate_p_and_l(original_premium)
        #
        #
        # # Market same, vol up (top center)
        # price, delta, gamma, theta, vega = calculate_call_data(F, K, T, r, sigma,
        #                                                        vol_delta=vol_delta) if option_type == 'call' else calculate_put_data(
        #     F, K, T, r, sigma, vol_delta=vol_delta)
        # premium = price * quantity
        # same_up = Portfolio(premium, delta, gamma, theta, vega, quantity)
        # same_up.calculate_p_and_l(original_premium)
        #
        #
        # # Market same, vol down (bottom center)
        # price, delta, gamma, theta, vega = calculate_call_data(F, K, T, r, sigma,
        #                                                        vol_delta=-vol_delta) if option_type == 'call' else calculate_put_data(
        #     F, K, T, r, sigma, vol_delta=-vol_delta)
        # premium = price * quantity
        # same_down = Portfolio(premium, delta, gamma, theta, vega, quantity)
        # same_down.calculate_p_and_l(original_premium)
        #
        #
        # # Create (inverted) cell text for the heatmap using Portfolio  objects
        # cell_text = [
        #     [down_down.to_plotly_format(), same_down.to_plotly_format(), up_down.to_plotly_format()],
        #     [down_same.to_plotly_format(), same_same.to_plotly_format(), up_same.to_plotly_format()],
        #     [down_up.to_plotly_format(), same_up.to_plotly_format(), up_up.to_plotly_format()]
        # ]
        # p_and_l_data = [
        #     [down_down.p_and_l, same_down.p_and_l, up_down.p_and_l],
        #     [down_same.p_and_l, same_same.p_and_l, up_same.p_and_l],
        #     [down_up.p_and_l, same_up.p_and_l, up_up.p_and_l]
        # ]

        # Create a grid of data
        fig = go.Figure(data=go.Heatmap(
            z=p_and_l_data,
            text=cell_text,
            hoverinfo='text',
            texttemplate="%{text}",
            x=[f'Down ${market_delta}', 'Same', f'Up ${market_delta}'],
            y=[f'Down {vol_delta}%', 'Same', f'Up {vol_delta}%'],
            colorscale='RdYlGn'
        ))
        # Add axis labels
        fig.update_layout(
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True),
            xaxis_title="Market",
            yaxis_title="Volatility"

        )
        return fig

    except Exception as e:
        return go.Figure()


if __name__ == '__main__':
    app.run(debug=True)
