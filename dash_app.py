import numpy as np
import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.graph_objects as go
from portfolio import Portfolio
from black_scholes import calculate_call_data, calculate_put_data
from trade_parser import parse_structure
from vol_solver import interpolate_vol_from_delta, get_atm_volatility

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
        html.Label('Structure', style={'margin': '5px'}),
        dcc.Input(id='input-structure',
                  type='text',
                  value='CLZ5',
                  placeholder='Input Trade Structure (e.g CLZ5)',
                  style={'margin': '5px'}),
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



        html.Label('r (Risk-Free Rate)',
                   style={'margin': '5px'}),
        dcc.Input(id='input-r',
                  type='number',
                  value=DEFAULT_RISK_FREE_RATE,
                  min=0,
                  step=0.01,
                  placeholder='r (Risk-Free Rate)',
                  style={'margin': '5px'}),

    ], style={'display': 'flex', 'flexDirection': 'column'}),  # Flexbox for horizontal layout

    html.Hr(),


    html.Label('T (Time to Maturity [in Years])', style={'margin': '5px'}),
    dcc.Input(id='input-T',
              type='number',
              # value=DEFAULT_TIME_TO_MATURITY,
              # min=0,
              # step=0.01,
              placeholder='T (Time to Maturity)',
              disabled=True,
              style={'margin': '5px'}),
    html.Label('σ (Volatility)',
               style={'margin': '5px'}),
    dcc.Input(id='input-sigma',
              type='number',
              #value=DEFAULT_VOLATILITY,
              min=0,
              step=0.01,
              placeholder='σ (Volatility)',
              disabled=True,
              style={'margin': '5px'}),
    html.Label('Commodity Structure', style={'margin': '5px'}),
    dcc.Input(id='input-commodity',
              type='text',
              # value=DEFAULT_VOLATILITY,
              min=0,
              step=0.01,
              placeholder='Commodity',
              disabled=True,
              style={'margin': '5px'}),
    html.Label('Expiration', style={'margin': '5px'}),
    dcc.Input(id='input-expiration',
              type='text',
              # value=DEFAULT_VOLATILITY,
              min=0,
              step=0.01,
              placeholder='Expiration',
              disabled=True,
              style={'margin': '5px'}),

    dcc.Graph(
        id='heatmap',
        figure={},
        style={'height': HEATMAP_HEIGHT, 'width': '100%'}
    )
]
# @callback(
#     Output('input-T', 'value'),
#     Output('input-sigma', 'value'),
#     Output('input-commodity', 'value'),
#     Output('input-expiration', 'value'),
#     Input('input-structure', 'value'),
#     Input('option-type', 'value'),
#     Input('input-F', 'value'),
#     Input('input-K', 'value'),
#     Input('input-r', 'value')
# )
def parse_struct(structure, option_type, F, K, r):

    commodity, expiration, T = parse_structure(structure)
    volatility_matrix = pd.read_csv('Data/strike_matrix.csv', index_col=0)

    sigma = get_atm_volatility(volatility_matrix, expiration)  # Volatility from the matrix
    sigma = sigma / 100  # Convert percentage to decimal
    portfolio = calculate_scenario(F, K, T, r, sigma, option_type, 0, 0)
    delta = portfolio.delta
    new_vol = interpolate_vol_from_delta(volatility_matrix, expiration, delta, option_type)
    new_vol = new_vol / 100  # Convert percentage to decimal

    # Now we can use the interpolated volatility to create a new portfolio
    #portfolio2 = calculate_scenario(F, K, T, r, new_vol, option_type, 0, 0)
    sigma = new_vol
    return T, sigma, commodity, expiration

@callback(
    Output('input-T', 'value'),
    Output('input-sigma', 'value'),
    Output('input-commodity', 'value'),
    Output('input-expiration', 'value'),
    Output('heatmap', 'figure'),
    Input('input-structure', 'value'),
    Input('input-F', 'value'),
    Input('input-K', 'value'),
#    State('input-T', 'value'),
    Input('input-r', 'value'),
#    State('input-sigma', 'value'),
    Input('option-type', 'value'),
#    Input('calculate-button', 'n_clicks'),
    Input('vol_delta', 'value'),
    Input('market_delta', 'value'),
    Input('quantity', 'value')
)
def update_graph(structure, F, K, r, option_type, vol_delta, market_delta, quantity):
    try:

        T, sigma, commodity, expiration = parse_struct(structure, option_type, F, K, r)

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
        #return fig
        return T, sigma, commodity, expiration, fig

    except Exception as e:
        return None, None, None, None, go.Figure()


if __name__ == '__main__':
    app.run(debug=True)
