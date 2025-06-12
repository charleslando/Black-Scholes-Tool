import numpy as np
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
from portfolio import Portfolio
from black_scholes import calculate_call_data, calculate_put_data


def format_plot_data(price, delta, gamma, theta, vega, p_and_l=0):
    # Placeholder for data formatting logic
    return f"Price: {price:.2f}<br>Delta: {delta:.2f}<br>Gamma: {gamma:.2f}<br>Theta: {theta:.2f}<br>Vega: {vega:.2f}<br>P&L: {p_and_l:.2f}"




app = Dash()

app.layout = [
    html.H1(children='Welcome to the Black-Scholes Tool'),
    html.Div(children='This tool allows you to calculate Black-Scholes option pricing.'),

    html.Label('Adjust Volatility'),
    dcc.Input(
        id='vol_delta',
        type='number',
        value=5,
        min=0,
        #max=100,
        step=1,
        style={'margin': '20px'}
    ),

    html.Label('Adjust Market'),
    dcc.Input(
        id='market_delta',
        type='number',
        value=5,
        min=0,
        #max=100,
        step=1,
        style={'margin': '20px'}
    ),
    html.Label("Quantity"),
    dcc.Input(
        id='quantity',
        type='number',
        value=1,
        min=1,
        step=1),

    html.Button('Calculate', id='calculate-button', style={'margin': '10px', 'padding': '10px 20px',
                                                           'backgroundColor': '#4CAF50'}),
    dcc.RadioItems(options=['put', 'call'], value='call', id='option-type'),

    html.Hr(),

    html.Div(children='Input Parameters for Black-Scholes Calculation:'),
    html.Div([
        html.Label('F (Forward Price)', style={'margin': '5px'}),
        dcc.Input(id='input-F', type='number', placeholder='F (Forward Price)', style={'margin': '5px'}),

        html.Label('K (Strike Price)', style={'margin': '5px'}),
        dcc.Input(id='input-K', type='number', placeholder='K (Strike Price)', style={'margin': '5px'}),

        html.Label('T (Time to Maturity)', style={'margin': '5px'}),
        dcc.Input(id='input-T', type='number', placeholder='T (Time to Maturity)', style={'margin': '5px'}),

        html.Label('r (Risk-Free Rate)', style={'margin': '5px'}),
        dcc.Input(id='input-r', type='number', placeholder='r (Risk-Free Rate)', style={'margin': '5px'}),

        html.Label('σ (Volatility)', style={'margin': '5px'}),
        dcc.Input(id='input-sigma', type='number', placeholder='σ (Volatility)', style={'margin': '5px'})

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
    # Create a grid of data for plotting

    if (None in [F, K, T, r, sigma]) or n_clicks is None:
        return go.Figure()

    # Calculate same_same (center - no changes)
    original_price, delta, gamma, theta, vega = calculate_call_data(F, K, T, r,
                                                                    sigma) if option_type == 'call' else calculate_put_data(
        F, K, T, r, sigma)
    same_same = Portfolio(original_price, delta, gamma, theta, vega, quantity)
    same_same.calculate_p_and_l(original_price)

    # Market up, vol same (center right)
    price, delta, gamma, theta, vega = calculate_call_data(F, K, T, r, sigma,
                                                           market_delta=market_delta) if option_type == 'call' else calculate_put_data(
        F, K, T, r, sigma, market_delta=market_delta)
    up_same = Portfolio(price, delta, gamma, theta, vega, quantity)
    up_same.calculate_p_and_l(original_price)


    # Market down, vol same (center left)
    price, delta, gamma, theta, vega = calculate_call_data(F, K, T, r, sigma,
                                                           market_delta=-market_delta) if option_type == 'call' else calculate_put_data(
        F, K, T, r, sigma, market_delta=-market_delta)
    down_same = Portfolio(price, delta, gamma, theta, vega, quantity)
    down_same.calculate_p_and_l(original_price)


    # Market up, vol up (top right)
    price, delta, gamma, theta, vega = calculate_call_data(F, K, T, r, sigma, vol_delta=vol_delta,
                                                           market_delta=market_delta) if option_type == 'call' else calculate_put_data(
        F, K, T, r, sigma, vol_delta=vol_delta, market_delta=market_delta)
    up_up = Portfolio(price, delta, gamma, theta, vega, quantity)
    up_up.calculate_p_and_l(original_price)


    # Market down, vol down (bottom left)
    price, delta, gamma, theta, vega = calculate_call_data(F, K, T, r, sigma, vol_delta=-vol_delta,
                                                           market_delta=-market_delta) if option_type == 'call' else calculate_put_data(
        F, K, T, r, sigma, vol_delta=-vol_delta, market_delta=-market_delta)
    down_down = Portfolio(price, delta, gamma, theta, vega, quantity)
    down_down.calculate_p_and_l(original_price)


    # Market down, vol up (top left)
    price, delta, gamma, theta, vega = calculate_call_data(F, K, T, r, sigma, vol_delta=vol_delta,
                                                           market_delta=-market_delta) if option_type == 'call' else calculate_put_data(
        F, K, T, r, sigma, vol_delta=vol_delta, market_delta=-market_delta)
    down_up = Portfolio(price, delta, gamma, theta, vega, quantity)
    down_up.calculate_p_and_l(original_price)


    # Market up, vol down (bottom right)
    price, delta, gamma, theta, vega = calculate_call_data(F, K, T, r, sigma, vol_delta=-vol_delta,
                                                           market_delta=market_delta) if option_type == 'call' else calculate_put_data(
        F, K, T, r, sigma, vol_delta=-vol_delta, market_delta=market_delta)
    up_down = Portfolio(price, delta, gamma, theta, vega, quantity)
    up_down.calculate_p_and_l(original_price)


    # Market same, vol up (top center)
    price, delta, gamma, theta, vega = calculate_call_data(F, K, T, r, sigma,
                                                           vol_delta=vol_delta) if option_type == 'call' else calculate_put_data(
        F, K, T, r, sigma, vol_delta=vol_delta)
    same_up = Portfolio(price, delta, gamma, theta, vega, quantity)
    same_up.calculate_p_and_l(original_price)


    # Market same, vol down (bottom center)
    price, delta, gamma, theta, vega = calculate_call_data(F, K, T, r, sigma,
                                                           vol_delta=-vol_delta) if option_type == 'call' else calculate_put_data(
        F, K, T, r, sigma, vol_delta=-vol_delta)
    same_down = Portfolio(price, delta, gamma, theta, vega, quantity)
    same_down.calculate_p_and_l(original_price)


    # Create (inverted) cell text for the heatmap using Portfolio  objects
    cell_text = [
        [down_down.to_plotly_format(), same_down.to_plotly_format(), up_down.to_plotly_format()],
        [down_same.to_plotly_format(), same_same.to_plotly_format(), up_same.to_plotly_format()],
        [down_up.to_plotly_format(), same_up.to_plotly_format(), up_up.to_plotly_format()]
    ]
    p_and_l_data = [
        [down_down.p_and_l, same_down.p_and_l, up_down.p_and_l],
        [down_same.p_and_l, same_same.p_and_l, up_same.p_and_l],
        [down_up.p_and_l, same_up.p_and_l, up_up.p_and_l]
    ]

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
        xaxis_title="Market",
        yaxis_title="Volatility"
    )
    return fig


if __name__ == '__main__':
    app.run(debug=True)