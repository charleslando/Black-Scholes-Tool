from blackscholes import Black76Call, Black76Put
from portfolio import Portfolio

def format_data(price, delta, gamma, theta, vega, p_and_l= 0):
    # Placeholder for data formatting logic
    return "${:.2f} | Δ: {:.2f} | Γ: {:.2f} | Θ: {:.2f} | Vega: {:.2f} | P&L: {:.2f}".format(price, delta, gamma, theta, vega, p_and_l)


def plot_original_grid():
    # data = np.array([[-4, -3, -2], [-1, 0, 1], [2, 3, 4]])
    # fig, ax = plt.subplots()
    # cax = ax.imshow(data, cmap='RdYlGn', interpolation='nearest')
    # ax.set_xticks(np.arange(data.shape[1]))
    # ax.set_yticks(np.arange(data.shape[0]))
    # ax.set_xticklabels(['Goes Down', 'Stays Same', 'Goes Up'])
    # ax.set_yticklabels(['Goes Up', 'Stays Same', 'Goes Down'])
    # plt.colorbar(cax)
    # plt.xlabel('Market')
    # plt.ylabel('Volatility')
    # plt.title('Black Scholes Prediction')
    # plt.show()
    return

def calculate_call_data(F, K, T, r, sigma, vol_delta = 0, market_delta = 0):
    if vol_delta is None or abs(vol_delta) < 0:
        vol_delta = 0
    if market_delta is None or abs(market_delta) < 0:
        market_delta = 0
    # Convert deltas to percentages
    vol_delta_as_percent = vol_delta / 100

    F = F + market_delta
    sigma = sigma + vol_delta_as_percent

    call = Black76Call(F=F, K=K, T=T, r=r, sigma=sigma)
    call_price = call.price()
    call_delta = call.delta()
    call_vega  = call.vega()
    call_gamma = call.gamma()
    call_theta = call.theta()

    return call_price, call_delta, call_gamma, call_theta, call_vega

def calculate_put_data(F, K, T, r, sigma, vol_delta = 0, market_delta = 0):

    if vol_delta is None or abs(vol_delta) < 0:
        vol_delta = 0
    if market_delta is None or abs(market_delta) < 0:
        market_delta = 0
    # Convert deltas to percentages
    vol_delta_as_percent = vol_delta / 100

    F = F + market_delta
    sigma = sigma + vol_delta_as_percent

    put = Black76Put(F=F, K=K, T=T, r=r, sigma=sigma)
    put_price = put.price()
    put_delta = put.delta()
    put_gamma = put.gamma()
    put_theta = put.theta()
    put_vega = put.vega()
    return put_price, put_delta, put_gamma, put_theta, put_vega



# Example usage
if __name__ == "__main__":
    print("Black-Scholes Test")
    F = 100  # Forward price
    K = 100  # Strike price
    T = 1    # Time to maturity in years
    r = 0.035 # Risk-free interest rate
    sigma = 0.35  # Volatility

    vol_delta = 5
    market_delta = 5
    quantity = 1

    option_type = 'call'  # or 'put'

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

    print("Cell Text:")
    for row in cell_text:
        print(row)





