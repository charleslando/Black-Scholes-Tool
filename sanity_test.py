from black_scholes import calculate_call_data, calculate_put_data, format_data
F = 75
K = 57
T = .189
r = 0.05
sigma = .285

original_price, original_delta, original_gamma, original_theta, original_vega = calculate_put_data(F, K, T, r, sigma)
print(format_data(original_price, original_delta, original_gamma, original_theta, original_vega))

# #case: market stays the same and volatility goes up-> price should go up
# print("Market stays the same, volatility goes up 5 points")
# #sigma = 0.4
# price, delta, gamma, theta, vega = calculate_call_data(F, K, T, r, sigma)
# print(format_data(price, delta, gamma, theta, vega))
# print("\nMY PRICE:\n")
# #sigma = 0.35
# my_price, my_delta, my_gamma, my_theta, my_vega = calculate_call_data(F, K, T, r, sigma, vol_delta=5)
# print(format_data(my_price, my_delta, my_gamma, my_theta, my_vega))

