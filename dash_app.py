import numpy as np
import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input, State, ALL, MATCH, callback_context
import plotly.graph_objects as go
from portfolio import Portfolio
from black_scholes import calculate_call_data, calculate_put_data
from trade_parser import parse_structure
from vol_solver import get_atm_volatility, interpolate_vol_from_strike

# Constants
DEFAULT_FORWARD_PRICE = 65.5
DEFAULT_STRIKE_PRICE = 65.5
DEFAULT_RISK_FREE_RATE = 0.05
DEFAULT_VOL_DELTA = 5
DEFAULT_MARKET_DELTA = 10
DEFAULT_QUANTITY = 1000
CONTRACT_BARRELS = 1000
HEATMAP_HEIGHT = '100vh'


class Leg:
    """
    Represents a single option leg in a strategy
    """

    def __init__(self, leg_id, structure, strike, quantity, option_type, action, forward_price):
        self.leg_id = leg_id
        self.structure = structure  # e.g., 'CLZ5'
        self.strike = strike
        self.quantity = quantity
        self.option_type = option_type  # 'call' or 'put'
        self.action = action  # 1 for buy, -1 for sell
        self.forward_price = forward_price

        # Parsed structure data
        self.commodity = None
        self.time_to_maturity = None
        self.expiration = None
        self.days_to_expiration = None

        # Greeks and pricing data
        self.price = 0
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        self.vega = 0
        self.premium = 0

        self._parse_structure()

    def _parse_structure(self):
        """Parse the structure to get commodity, expiration, etc."""
        try:
            self.commodity, self.time_to_maturity, self.expiration, self.days_to_expiration = parse_structure(
                self.structure)
        except Exception as e:
            print(f"Error parsing structure {self.structure}: {e}")

    def calculate_greeks(self, volatility, risk_free_rate, vol_delta=0, market_delta=0):
        """
        Calculate Greeks for this leg
        """
        adjusted_forward = self.forward_price + market_delta

        if self.option_type == 'call':
            price, delta, gamma, theta, vega = calculate_call_data(
                adjusted_forward, self.strike, self.time_to_maturity,
                risk_free_rate, volatility, vol_delta=vol_delta, market_delta=0
            )
        else:
            price, delta, gamma, theta, vega = calculate_put_data(
                adjusted_forward, self.strike, self.time_to_maturity,
                risk_free_rate, volatility, vol_delta=vol_delta, market_delta=0
            )

        # Apply quantity and action (buy/sell) multipliers
        multiplier = self.quantity * self.action

        self.price = price
        self.delta = delta * multiplier
        self.gamma = gamma * multiplier
        self.theta = theta * multiplier
        self.vega = vega * multiplier * 10

        # Calculate premium (price * quantity * barrels_per_contract * action)
        barrel_quantity = self.quantity * CONTRACT_BARRELS
        self.premium = price * barrel_quantity * self.action

    def get_volatility(self, strike_volatility_matrix, market_delta=0):
        """
        Get volatility for this leg based on strike and market conditions
        """
        return get_vol_from_strike(
            self.forward_price, self.strike, self.time_to_maturity,
            self.expiration, self.option_type, 0.05,  # Using default risk-free rate
            strike_volatility_matrix, market_delta=market_delta
        )

    def to_dict(self):
        """Convert leg to dictionary for easy serialization"""
        return {
            'leg_id': self.leg_id,
            'structure': self.structure,
            'strike': self.strike,
            'quantity': self.quantity,
            'option_type': self.option_type,
            'action': self.action,
            'forward_price': self.forward_price
        }


class StrategyPortfolio:
    """
    Represents a portfolio of option legs
    """

    def __init__(self, forward_price):
        self.legs = {}
        self.forward_price = forward_price
        self.total_portfolio = None

    def add_leg(self, leg):
        """Add a leg to the strategy"""
        self.legs[leg.leg_id] = leg

    def remove_leg(self, leg_id):
        """Remove a leg from the strategy"""
        if leg_id in self.legs:
            del self.legs[leg_id]

    def update_leg(self, leg_id, **kwargs):
        """Update a leg's parameters"""
        if leg_id in self.legs:
            leg = self.legs[leg_id]
            for key, value in kwargs.items():
                if hasattr(leg, key):
                    setattr(leg, key, value)

    def calculate_portfolio(self, strike_volatility_matrix, risk_free_rate=0.05, vol_delta=0, market_delta=0):
        if not self.legs:
            return None

        total_delta = total_gamma = total_theta = total_vega = total_premium = 0.0
        portfolio_price = 0.0

        # Calculate Greeks for all legs first
        for leg in self.legs.values():
            vol = leg.get_volatility(strike_volatility_matrix, market_delta)
            leg.calculate_greeks(vol, risk_free_rate, vol_delta, market_delta)

            total_delta += leg.delta
            total_gamma += leg.gamma
            total_theta += leg.theta
            total_vega += leg.vega
            total_premium += leg.premium

        # For single leg
        if len(self.legs) == 1:
            leg = next(iter(self.legs.values()))
            portfolio_price = leg.price
        else:
            # Multi-leg: simple ratio calculation
            # Find smallest quantity to get base ratio
            min_qty = min(abs(leg.quantity) for leg in self.legs.values())

            for leg in self.legs.values():
                ratio = abs(leg.quantity) // min_qty
                portfolio_price += ratio * leg.price * leg.action

            #portfolio_price = abs(portfolio_price)

        self.total_portfolio = Portfolio(
            portfolio_price, total_delta, total_gamma, total_theta, total_vega, total_premium
        )
        return self.total_portfolio

    def get_leg_count(self):
        """Get the number of legs in the strategy"""
        return len(self.legs)


def get_vol_from_strike(F, K, T, expiration, option_type, r, volatility_matrix, market_delta=0):
    """
    Get volatility based on relative moneyness (strike vs forward) rather than absolute strike.
    """
    # Adjust forward price based on market delta
    adjusted_forward = F + market_delta

    # Calculate the relative moneyness (difference between strike and forward)
    moneyness = K - adjusted_forward

    # Find the equivalent strike in the volatility matrix based on moneyness
    reference_forward = F
    equivalent_strike = reference_forward + moneyness

    # Get volatility based on the equivalent strike
    sigma = interpolate_vol_from_strike(volatility_matrix, expiration, equivalent_strike, option_type)
    sigma = sigma / 100  # Convert percentage to decimal

    return sigma


# Global strategy portfolio
strategy_portfolio = StrategyPortfolio(DEFAULT_FORWARD_PRICE)

app = Dash(__name__)
server = app.server


def create_leg_input_div(leg_id):
    """Create input components for a single leg"""
    return html.Div([
        html.H4(f"Leg {leg_id}", style={'color': '#2c3e50'}),
        html.Div([
            html.Label('Structure:', style={'margin-right': '10px', 'font-weight': 'bold'}),
            dcc.Input(
                id={'type': 'leg-structure', 'index': leg_id},
                type='text',
                value='CLZ5',
                style={'margin': '5px', 'width': '80px'}
            ),

            html.Label('Strike:', style={'margin-right': '10px', 'font-weight': 'bold'}),
            dcc.Input(
                id={'type': 'leg-strike', 'index': leg_id},
                type='number',
                value=DEFAULT_STRIKE_PRICE,
                step=0.1,
                style={'margin': '5px', 'width': '80px'}
            ),

            html.Label('Quantity:', style={'margin-right': '10px', 'font-weight': 'bold'}),
            dcc.Input(
                id={'type': 'leg-quantity', 'index': leg_id},
                type='number',
                value=DEFAULT_QUANTITY,
                min=1,
                step=1,
                style={'margin': '5px', 'width': '80px'}
            ),

            html.Label('Type:', style={'margin-right': '10px', 'font-weight': 'bold'}),
            dcc.Dropdown(
                id={'type': 'leg-option-type', 'index': leg_id},
                options=[
                    {'label': 'Call', 'value': 'call'},
                    {'label': 'Put', 'value': 'put'}
                ],
                value='call',
                style={'margin': '5px', 'width': '80px'}
            ),

            html.Label('Action:', style={'margin-right': '10px', 'font-weight': 'bold'}),
            dcc.Dropdown(
                id={'type': 'leg-action', 'index': leg_id},
                options=[
                    {'label': 'Buy', 'value': 1},
                    {'label': 'Sell', 'value': -1}
                ],
                value=1,
                style={'margin': '5px', 'width': '80px'}
            ),

            html.Button(
                f'Remove Leg {leg_id}',
                id={'type': 'remove-leg', 'index': leg_id},
                style={
                    'margin': '5px',
                    'backgroundColor': '#e74c3c',
                    'color': 'white',
                    'border': 'none',
                    'padding': '5px 10px',
                    'cursor': 'pointer'
                }
            )
        ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '10px'})
    ], style={
        'border': '1px solid #bdc3c7',
        'padding': '15px',
        'margin': '10px 0',
        'borderRadius': '5px',
        'backgroundColor': '#f8f9fa'
    })


app.layout = html.Div([
    html.H1(children='Advanced Black-Scholes Multi-Leg Tool', style={'textAlign': 'center', 'color': '#2c3e50'}),
    html.Div(children='Create complex option strategies with multiple legs.',
             style={'textAlign': 'center', 'margin-bottom': '20px'}),

    html.Hr(),

    # Global Parameters Section
    html.Div([
        html.H3('Global Parameters', style={'color': '#34495e'}),
        html.Div([
            html.Label('Forward Price (F):', style={'margin-right': '10px', 'font-weight': 'bold'}),
            dcc.Input(
                id='global-forward-price',
                type='number',
                value=DEFAULT_FORWARD_PRICE,
                min=0,
                step=0.01,
                style={'margin': '5px', 'width': '100px'}
            ),

            html.Label('Risk-Free Rate (r):',
                       style={'margin-left': '20px', 'margin-right': '10px', 'font-weight': 'bold'}),
            dcc.Input(
                id='global-risk-free-rate',
                type='number',
                value=DEFAULT_RISK_FREE_RATE,
                min=0,
                step=0.01,
                style={'margin': '5px', 'width': '100px'}
            )
        ], style={'display': 'flex', 'alignItems': 'center', 'margin': '10px 0'})
    ]),

    html.Hr(),

    # Legs Section
    html.Div([
        html.H3('Strategy Legs', style={'color': '#34495e'}),
        html.Button(
            'Add New Leg',
            id='add-leg-button',
            style={
                'backgroundColor': '#27ae60',
                'color': 'white',
                'border': 'none',
                'padding': '10px 20px',
                'cursor': 'pointer',
                'borderRadius': '5px',
                'margin': '10px 0'
            }
        ),
        html.Div(id='legs-container', children=[create_leg_input_div(1)])
    ]),

    html.Hr(),

    # Scenario Analysis Parameters
    html.Div([
        html.H3('Scenario Analysis', style={'color': '#34495e'}),
        html.Div([
            html.Label('Grid Size:', style={'margin-right': '10px', 'font-weight': 'bold'}),
            dcc.Dropdown(
                id='grid-size',
                options=[
                    {'label': '3x3', 'value': 3},
                    {'label': '5x5', 'value': 5},
                    {'label': '7x7', 'value': 7},
                    {'label': '9x9', 'value': 9}
                ],
                value=3,
                style={'width': '100px', 'margin-right': '20px'}
            ),

            html.Label('Vol Delta (%):', style={'margin-right': '10px', 'font-weight': 'bold'}),
            dcc.Input(
                id='vol-delta',
                type='number',
                value=DEFAULT_VOL_DELTA,
                min=0,
                step=1,
                style={'margin-right': '20px', 'width': '80px'}
            ),

            html.Label('Market Delta ($):', style={'margin-right': '10px', 'font-weight': 'bold'}),
            dcc.Input(
                id='market-delta',
                type='number',
                value=DEFAULT_MARKET_DELTA,
                min=0,
                step=1,
                style={'width': '80px'}
            )
        ], style={'display': 'flex', 'alignItems': 'center', 'margin': '10px 0'})
    ]),

    html.Hr(),

    # Results Section
    dcc.Store(id='leg-counter', data=1),  # Store to track leg counter
    dcc.Graph(
        id='heatmap',
        figure={},
        style={'height': HEATMAP_HEIGHT, 'width': '100%'}
    )
])


@callback(
    Output('legs-container', 'children'),
    Output('leg-counter', 'data'),
    Input('add-leg-button', 'n_clicks'),
    Input({'type': 'remove-leg', 'index': ALL}, 'n_clicks'),
    State('legs-container', 'children'),
    State('leg-counter', 'data'),
    prevent_initial_call=False  # Allow initial call to create the first leg
)
def manage_legs(add_clicks, remove_clicks, current_legs, leg_counter):
    """Add or remove legs dynamically"""
    ctx = callback_context

    # Initialize if no legs exist
    if not current_legs:
        return [create_leg_input_div(1)], 1

    if not ctx.triggered:
        return current_legs, leg_counter

    trigger_id = ctx.triggered[0]['prop_id']

    if 'add-leg-button' in trigger_id and add_clicks:
        # Add new leg
        new_leg_id = leg_counter + 1
        new_leg_div = create_leg_input_div(new_leg_id)
        current_legs.append(new_leg_div)
        return current_legs, new_leg_id

    elif 'remove-leg' in trigger_id and any(remove_clicks):
        # Remove leg - extract leg_id from the trigger
        import json
        trigger_dict = json.loads(trigger_id.split('.')[0])
        leg_id_to_remove = trigger_dict['index']

        # Filter out the leg to remove
        updated_legs = []
        for leg_div in current_legs:
            # Check if this is the leg to remove by looking at the leg_id in the H4 element
            if leg_div['props']['children'][0]['props']['children'] != f"Leg {leg_id_to_remove}":
                updated_legs.append(leg_div)

        # Ensure at least one leg remains
        if not updated_legs:
            updated_legs = [create_leg_input_div(1)]
            leg_counter = 1

        return updated_legs, leg_counter

    return current_legs, leg_counter


@callback(
    Output('heatmap', 'figure'),
    Input('global-forward-price', 'value'),
    Input('global-risk-free-rate', 'value'),
    Input('grid-size', 'value'),
    Input('vol-delta', 'value'),
    Input('market-delta', 'value'),
    Input({'type': 'leg-structure', 'index': ALL}, 'value'),
    Input({'type': 'leg-strike', 'index': ALL}, 'value'),
    Input({'type': 'leg-quantity', 'index': ALL}, 'value'),
    Input({'type': 'leg-option-type', 'index': ALL}, 'value'),
    Input({'type': 'leg-action', 'index': ALL}, 'value'),
    State('legs-container', 'children'),
    prevent_initial_call=False  # Allow initial call to show graph
)
def update_heatmap(forward_price, risk_free_rate, grid_size, vol_delta, market_delta,
                   leg_structures, leg_strikes, leg_quantities, leg_option_types, leg_actions,
                   legs_container):
    """Update the heatmap based on current strategy configuration"""

    try:
        # Create a basic figure if inputs are not ready
        if not all([forward_price, risk_free_rate]) or not any([leg_structures, leg_strikes,
                                                                leg_quantities, leg_option_types, leg_actions]):
            fig = go.Figure()
            fig.update_layout(
                title='Configure legs to see portfolio analysis',
                xaxis_title="Market",
                yaxis_title="Volatility",
                height=600
            )
            return fig

        # Clear existing strategy
        strategy_portfolio.legs.clear()
        strategy_portfolio.forward_price = forward_price

        # Load strike volatility matrix
        try:
            strike_volatility_matrix = pd.read_csv('Data/strike_vol_matrix.csv', index_col=0)
        except FileNotFoundError:
            # Create dummy data if file doesn't exist for testing
            print("Warning: strike_vol_matrix.csv not found, using dummy data")
            dates = pd.date_range('2024-01-01', periods=12, freq='M')
            strikes = np.arange(50, 81, 5)  # 50, 55, 60, 65, 70, 75, 80
            strike_volatility_matrix = pd.DataFrame(
                np.random.uniform(20, 40, (len(dates), len(strikes))),
                index=dates.strftime('%Y-%m'),
                columns=strikes
            )

        # Create legs from inputs
        legs_created = 0
        for i, (structure, strike, quantity, option_type, action) in enumerate(
                zip(leg_structures or [], leg_strikes or [], leg_quantities or [],
                    leg_option_types or [], leg_actions or [])
        ):
            if all([structure, strike is not None, quantity is not None, option_type, action is not None]):
                try:
                    leg = Leg(
                        leg_id=i + 1,
                        structure=structure,
                        strike=strike,
                        quantity=quantity,
                        option_type=option_type,
                        action=action,
                        forward_price=forward_price
                    )
                    strategy_portfolio.add_leg(leg)
                    legs_created += 1
                except Exception as e:
                    print(f"Error creating leg {i + 1}: {e}")

        if legs_created == 0:
            fig = go.Figure()
            fig.update_layout(
                title='No valid legs configured',
                xaxis_title="Market",
                yaxis_title="Volatility",
                height=600
            )
            return fig

        # Create scenario grids
        vol_levels = np.linspace(-vol_delta, vol_delta, grid_size)
        market_levels = np.linspace(-market_delta, market_delta, grid_size)

        # Calculate base portfolio for P&L comparison
        try:
            base_portfolio = strategy_portfolio.calculate_portfolio(
                strike_volatility_matrix, risk_free_rate, 0, 0
            )
            if base_portfolio is None:
                raise ValueError("Failed to calculate base portfolio")
            base_premium = base_portfolio.premium
        except Exception as e:
            print(f"Error calculating base portfolio: {e}")
            fig = go.Figure()
            fig.update_layout(
                title=f'Error calculating portfolio: {str(e)}',
                xaxis_title="Market",
                yaxis_title="Volatility",
                height=600
            )
            return fig

        # Calculate scenarios
        scenarios = {}
        cell_text = []
        p_and_l_data = []

        for i, vol_change in enumerate(vol_levels):
            row_text = []
            row_data = []

            for j, market_change in enumerate(market_levels):
                try:
                    portfolio = strategy_portfolio.calculate_portfolio(
                        strike_volatility_matrix, risk_free_rate, vol_change, market_change
                    )
                    portfolio.calculate_p_and_l(base_premium)

                    # Calculate volatilities for each leg
                    leg_vols = []
                    leg_debug_info = []

                    for leg_id, leg in strategy_portfolio.legs.items():
                        base_vol = leg.get_volatility(strike_volatility_matrix, market_delta=market_change)
                        effective_vol = base_vol + (vol_change / 100)
                        leg_vols.append(f"{effective_vol:.4f}")

                        # Debug info for each leg
                        action_str = "BUY" if leg.action == 1 else "SELL"
                        leg_debug_info.append(
                            f"L{leg_id}: {action_str} {leg.quantity} {leg.option_type.upper()} "
                            f"@{leg.strike} | P=${leg.price:.4f} | Δ={leg.delta:.0f} | "
                            f"Γ={leg.gamma:.2f} | θ={leg.theta:.2f} | ν={leg.vega:.0f}"
                        )

                    vol_text = f"VOLS: {' / '.join(leg_vols)}"
                    debug_text = "<br>".join(leg_debug_info)

                    scenarios[(vol_change, market_change)] = portfolio
                    cell_display = f"{portfolio.to_plotly_format()}<br>{vol_text}<br><br>LEG DEBUG:<br>{debug_text}"
                    #cell_display = f"{portfolio.to_plotly_format()}<br>{vol_text}<br>"
                    row_text.append(cell_display)
                    row_data.append(portfolio.p_and_l)

                except Exception as e:
                    print(f"Error calculating scenario ({vol_change}, {market_change}): {e}")
                    row_text.append(f"Error: {str(e)}")
                    row_data.append(0)

            cell_text.append(row_text)
            p_and_l_data.append(row_data)

        # Create axis labels
        x_labels = [f'MARKET AT {forward_price + market_change:.1f}' for market_change in market_levels]
        y_labels = []
        for vol_change in vol_levels:
            if abs(vol_change) < 0.01:
                y_labels.append('Same')
            elif vol_change > 0:
                y_labels.append(f'Up {vol_change:.1f}%')
            else:
                y_labels.append(f'Down {abs(vol_change):.1f}%')

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=p_and_l_data,
            text=cell_text,
            hoverinfo='text',
            texttemplate="%{text}",
            x=x_labels,
            y=y_labels,
            colorscale='RdYlGn',
            zmid = 0
        ))

        fig.update_layout(
            title=f'Portfolio Scenario Analysis',
            xaxis=dict(fixedrange=True, title="Market", side='top'),
            yaxis=dict(fixedrange=True, title="Volatility"),
            height=600
        )

        return fig

    except Exception as e:
        print(f"Error updating heatmap: {e}")
        import traceback
        traceback.print_exc()

        fig = go.Figure()
        fig.update_layout(
            title=f'Error: {str(e)}',
            xaxis_title="Market",
            yaxis_title="Volatility",
            height=600
        )
        return fig


if __name__ == '__main__':
    app.run(debug=True)