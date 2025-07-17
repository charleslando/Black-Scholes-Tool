import numpy as np
import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.graph_objects as go
from portfolio import Portfolio
from black_scholes import calculate_call_data, calculate_put_data
from trade_parser import parse_structure
from vol_solver import interpolate_vol_from_delta, get_atm_volatility, interpolate_vol_from_strike


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
    html.Div(children='This tool allows you to calculate and visualise Black-Scholes option pricing.'),

    html.Hr(),

    html.Div([
        html.Label('Grid Size'),
        dcc.Dropdown(
            id='grid-size',
            options=[
                {'label': '3x3', 'value': 3},
                {'label': '5x5', 'value': 5},
                {'label': '7x7', 'value': 7},
                {'label': '9x9', 'value': 9}
            ],
            value=3,
            style={'margin': '20px', 'width': '200px'}
        ),

        html.Label('Grid Format'),
        dcc.RadioItems(
            id='grid-format',
            options=[
                {'label': 'Multiplicative (e.g., -2x, -1x, 0, +1x, +2x)', 'value': 'multiplicative'},
                {'label': 'Linspace (evenly spaced between bounds)', 'value': 'linspace'}
            ],
            value='linspace',
            style={'margin': '20px'}
        ),
    ], style={'display': 'flex', 'alignItems': 'center'}),

    html.Div([
        html.Label('Table Type', style={'fontWeight': 'bold', 'margin': '10px'}),
        dcc.RadioItems(
            id='table-type',
            options=[
                {'label': 'Delta/Vol Table', 'value': 'delta_vol'},
                {'label': 'Strike/Vol Table', 'value': 'strike_vol'}
            ],
            value='delta_vol',
            style={'margin': '10px'}
        ),
    ]),

    html.Label('Adjust Volatility'),
    dcc.Input(
        id='vol_delta',
        type='number',
        value=DEFAULT_VOL_DELTA,
        min=0,
        # max=100,
        step=1,
        style={'margin': '20px'}
    ),

    html.Label('Adjust Market'),
    dcc.Input(
        id='market_delta',
        type='number',
        value=DEFAULT_MARKET_DELTA,
        min=0,
        # max=100,
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
                  step=.1,
                  placeholder='F (Forward Price)',
                  style={'margin': '5px'}),

        html.Label('K (Strike Price)', style={'margin': '5px'}),
        dcc.Input(id='input-K',
                  type='number',
                  value=DEFAULT_STRIKE_PRICE,
                  min=0,
                  step=.1,
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

    html.Label('T (Time to Maturity [in Days])', style={'margin': '5px'}),
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
              # value=DEFAULT_VOLATILITY,
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


def parse_struct(structure):
    commodity, T, expiration, days_to_expiration = parse_structure(structure)
    return commodity, T, expiration, days_to_expiration


def get_vol_from_delta(F, K, T, expiration, option_type, r, volatility_matrix):
    sigma = get_atm_volatility(volatility_matrix, expiration)  # Volatility from the matrix
    sigma = sigma / 100  # Convert percentage to decimal
    print(f"DEBUG: Initial sigma from matrix: {sigma}")
    portfolio = calculate_scenario(F, K, T, r, sigma, option_type, 0, 0)
    delta = portfolio.delta
    new_vol = interpolate_vol_from_delta(volatility_matrix, expiration, delta, option_type)
    new_vol = new_vol / 100  # Convert percentage to decimal
    # Now we can use the interpolated volatility to create a new portfolio
    # portfolio2 = calculate_scenario(F, K, T, r, new_vol, option_type, 0, 0)
    sigma = new_vol
    return sigma

def get_vol_from_strike(F, K, T, expiration, option_type, r, volatility_matrix):
    sigma = interpolate_vol_from_strike(volatility_matrix, expiration, K, option_type)
    sigma = sigma / 100  # Convert percentage to decimal
    return sigma


@callback(
    Output('input-T', 'value'),
    Output('input-sigma', 'value'),
    Output('input-commodity', 'value'),
    Output('input-expiration', 'value'),
    Output('heatmap', 'figure'),
    Input('table-type', 'value'),
    Input('input-structure', 'value'),
    Input('input-F', 'value'),
    Input('input-K', 'value'),
    Input('input-r', 'value'),
    Input('option-type', 'value'),
    Input('vol_delta', 'value'),
    Input('market_delta', 'value'),
    Input('quantity', 'value'),
    Input('grid-size', 'value'),
    Input('grid-format', 'value')
)
def update_graph(table_type, structure, F, K, r, option_type, vol_delta, market_delta, quantity, grid_size,
                 grid_format):
    try:
        #print(f"DEBUG: Inputs - structure: {structure}, F: {F}, K: {K}, r: {r}")

        # Check if required inputs are valid
        if not structure or F is None or K is None or r is None:
            print("DEBUG: Missing required inputs")
            return None, None, None, None, go.Figure()

        #print("DEBUG: Parsing structure...")
        commodity, T, expiration, days_to_expiration = parse_struct(structure)
        delta_volatility_matrix = pd.read_csv('Data/delta_vol_matrix.csv', index_col=0)
        strike_volatility_matrix = pd.read_csv('Data/strike_vol_matrix.csv', index_col=0)

        #print(f"DEBUG: Parsed - T: {T}, commodity: {commodity}, expiration: {expiration}")

        #get Vol
        if table_type == 'delta_vol':
            # Get volatility based on delta
            sigma = get_vol_from_delta(F, K, T, expiration, option_type, r, delta_volatility_matrix)
        else:
            sigma = get_vol_from_strike(F, K, T, expiration, option_type, r, strike_volatility_matrix)


        if sigma is None:
            #print("DEBUG: Failed to get volatility")
            return None, None, None, None, go.Figure()

        #print("DEBUG: Calculating original portfolio...")
        original_portfolio = calculate_scenario(F, K, T, r, sigma, option_type, 0, 0, quantity)
        original_premium = original_portfolio.price * quantity
        #print(f"DEBUG: Original premium: {original_premium}")

        # Create levels based on grid size and format
        if grid_format == 'multiplicative':
            half_size = grid_size // 2
            vol_levels = [i * vol_delta for i in range(-half_size, half_size + 1)]
            market_levels = [i * market_delta for i in range(-half_size, half_size + 1)]
        else:
            vol_levels = np.linspace(-vol_delta, vol_delta, grid_size)
            market_levels = np.linspace(-market_delta, market_delta, grid_size)

        #print(f"DEBUG: Vol levels: {vol_levels}")
        #print(f"DEBUG: Market levels: {market_levels}")

        # Calculate scenarios for all combinations
        scenarios.clear()
        #print("DEBUG: Calculating scenarios...")
        vols = []
        for i, vol_change in enumerate(vol_levels):
            for j, market_change in enumerate(market_levels):
                portfolio = calculate_scenario(F, K, T, r, sigma, option_type, vol_change, market_change, quantity)
                portfolio.calculate_p_and_l(original_premium)
                scenarios[(vol_change, market_change)] = portfolio
                vols.append(f"VOLATILITY: {sigma + (vol_change / 100)}")  # Store volatility for hover text
                #if i == 0 and j == 0:  # Log first scenario for debugging
                    #print(f"DEBUG: First scenario P&L: {portfolio.p_and_l}")
        print(f"\n\n{vols}\n\n")
        cell_text = []
        p_and_l_data = []

        # Create cell text for the heatmap using Portfolio objects
        for vol_change in vol_levels:
            row_text = []
            row_data = []
            #volatilities = []
            for market_change in market_levels:
                portfolio = scenarios[(vol_change, market_change)]
                row_text.append(portfolio.to_plotly_format())
                row_data.append(portfolio.p_and_l)
            cell_text.append(row_text)
            p_and_l_data.append(row_data)

        #print(f"DEBUG: P&L data shape: {len(p_and_l_data)}x{len(p_and_l_data[0]) if p_and_l_data else 0}")
        #print(f"DEBUG: Sample P&L values: {p_and_l_data[0][:3] if p_and_l_data else 'None'}")

        # Create labels for x and y axes based on format
        x_labels = []
        y_labels = []

        if grid_format == 'multiplicative':
            for market_change in market_levels:
                if market_change == 0:
                    x_labels.append('Same')
                elif market_change > 0:
                    if market_change == market_delta:
                        x_labels.append(f'Up ${market_delta}')
                    else:
                        multiplier = int(market_change / market_delta)
                        x_labels.append(f'Up ${multiplier}x{market_delta}')
                else:
                    if abs(market_change) == market_delta:
                        x_labels.append(f'Down ${market_delta}')
                    else:
                        multiplier = int(abs(market_change) / market_delta)
                        x_labels.append(f'Down ${multiplier}x{market_delta}')

            for vol_change in vol_levels:
                if vol_change == 0:
                    y_labels.append('Same')
                elif vol_change > 0:
                    if vol_change == vol_delta:
                        y_labels.append(f'Up {vol_delta}%')
                    else:
                        multiplier = int(vol_change / vol_delta)
                        y_labels.append(f'Up {multiplier}x{vol_delta}%')
                else:
                    if abs(vol_change) == vol_delta:
                        y_labels.append(f'Down {vol_delta}%')
                    else:
                        multiplier = int(abs(vol_change) / vol_delta)
                        y_labels.append(f'Down {multiplier}x{vol_delta}%')
        else:
            for market_change in market_levels:
                if abs(market_change) < 0.01:
                    x_labels.append('Same')
                elif market_change > 0:
                    x_labels.append(f'Up ${market_change:.1f}')
                else:
                    x_labels.append(f'Down ${abs(market_change):.1f}')

            for vol_change in vol_levels:
                if abs(vol_change) < 0.01:
                    y_labels.append('Same')
                elif vol_change > 0:
                    y_labels.append(f'Up {vol_change:.1f}%')
                else:
                    y_labels.append(f'Down {abs(vol_change):.1f}%')

        # print(f"DEBUG: X labels: {x_labels}")
        # print(f"DEBUG: Y labels: {y_labels}")

        # Create a grid of data
        #print("DEBUG: Creating heatmap...")
        fig = go.Figure(data=go.Heatmap(
            z=p_and_l_data,
            text=cell_text,
            #hoverinfo=vols,
            hoverinfo='text',
            texttemplate="%{text}",
            x=x_labels,
            y=y_labels,
            colorscale='RdYlGn'
        ))

        # Add axis labels
        fig.update_layout(
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True),
            xaxis_title="Market",
            yaxis_title="Volatility"
        )

        #print("DEBUG: Figure created successfully")
        return days_to_expiration, sigma, commodity, expiration, fig

    except Exception as e:
        #print(f"DEBUG: Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, go.Figure()



if __name__ == '__main__':
    app.run(debug=True)