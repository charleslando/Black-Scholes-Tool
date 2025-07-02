import pandas as pd
from dash_app import calculate_scenario


vol_matrix = pd.read_csv('volatility_matrix.csv', index_col=0)

call_delta_map = {
    '1DP': 0.99,

    '10DP': 0.90,
    '15DP': 0.85,
    '25DP': 0.75,
    '35DP': 0.65,
    '50D': 0.5,
    '35DC': 0.35,
    '25DC': 0.25,
    '15DC': 0.15,
    '10DC': 0.10,

    '1DC': 0.01,
}

put_delta_map = {
    '1DP': 0.01,

    '10DP': 0.10,
    '15DP': 0.15,
    '25DP': 0.25,
    '35DP': 0.35,
    '50D': 0.5,
    '35DC': 0.65,
    '25DC': 0.75,
    '15DC': 0.85,
    '10DC': 0.90,

    '1DC': 0.99,
}

def get_atm_volatility(vol_matrix, month):
    try:
        # label‐based lookup—super concise
        return vol_matrix.at[month, '50D']
    except KeyError:
        raise KeyError(f"Month {month!r} or column '50D' not found in vol_matrix.")


def interpolate_vol(df, month, delta, option_type):
    # Insert out of bounds deltas by adding weighted values
    weights = df['Weighting'].values.tolist()
    df.drop(columns=['Weighting'], inplace=True)

    ten_dp_vals = df['10DP'].values.tolist()
    one_dp_vals = [ten_dp + weight for ten_dp, weight in zip(ten_dp_vals, weights)]
    # Add 1DP column at the beginning
    vol_matrix.insert(0, '1DP', one_dp_vals)

    ten_dc_vals = df['10DC'].values.tolist()
    one_dc_vals = [ten_dc + weight for ten_dc, weight in zip(ten_dc_vals, weights)]
    # Add 1DC column at the end
    vol_matrix.insert(len(vol_matrix.columns), '1DC', one_dc_vals)



    # # Add 1DC column at the end
    # vol_matrix['1DC'] = vol_matrix['10DC'] + weighted_val



    #Filter relevant data columns
    if option_type.lower() == 'call':
       relevant = {k: v for k, v in call_delta_map.items()}
    else:
       relevant = {k: v for k, v in put_delta_map.items()}




    delta_vals = list(relevant.values())
    columns = list(relevant.keys())

    # weight_delta = 1
    # weight = df.at[month, 'Weighted']
    # if delta < min(delta_vals) or delta > max(delta_vals):
    #     if(delta < min(delta_vals)):
    #         #interpolate between 10 and weight_delta
    #         delta_vals.insert(delta)




    #find the two bracketing deltas
    for i in range(len(delta_vals) - 1):
        d1, d2 = delta_vals[i], delta_vals[i + 1]
        if d1 <= delta <= d2 or d2 <= delta <= d1:
            # Interpolate between the two deltas
            col1, col2 = columns[i], columns[i + 1]
            vol1 = df.at[month, col1]
            vol2 = df.at[month, col2]

            # Linear interpolation
            weight = (delta - d1) / (d2 - d1)
            interpolated_vol = vol1 + weight * (vol2 - vol1)
            return interpolated_vol




    raise ValueError(f"Delta {delta} not found in the specified month {month} for option type {option_type}.")
# def find_closest_col_idx(df, target, option_type):
#     # find the column label closest to target
#     closest_label = min(df.columns, key=lambda c: (abs(int(c[:2]) - target) if c[-1] == option_type else float('inf')))
#     closest_val = df.columns.get_loc(closest_label)
#     second_closest_label = 0
#     if (closest_label == 0) or (closest_label == len(df.columns) - 1):
#         second_closest_label = abs(closest_label-1)
#     else:
#         second_closest_label = min(df.columns.get_loc(closest_label) - 1, df.columns.get_loc(closest_label) + 1)
#
#     second_closest_val = df.columns.get_loc()
#
#
#
#
#     # #    return df.columns.get_loc(closest_label)
#     # second_closest = min(df.columns.get_loc(closest_label)
#     # get its integer position
#     return df.columns.get_loc(closest_label)


# constants for the Black-Scholes model

F = 100
K = 140
T = 0.5
r = 0.05
option_type = 'call'
month = 'Jan'
sigma = get_atm_volatility(vol_matrix, month)  # Volatility from the matrix
sigma = sigma / 100  # Convert percentage to decimal
portfolio = calculate_scenario(F, K, T, r, sigma, option_type, 0,0)

print(f"Portfolio Details:\n{portfolio.__str__()}")

delta = portfolio.delta

# col_idx = find_closest_col_idx(vol_matrix, delta, option_type)

print(f"Delta: {delta}\n")

new_vol = interpolate_vol(vol_matrix, month, delta, option_type)
new_vol = new_vol / 100  # Convert percentage to decimal
print(f"Interpolated Volatility: {new_vol}\n")

# Now we can use the interpolated volatility to create a new portfolio
portfolio2 = calculate_scenario(F, K, T, r, new_vol, option_type, 0, 0)
print(f"Portfolio Details with adjusted volatility:\n{portfolio2.__str__()}\n")

print(f"\n\nFINAL PRICE: {portfolio2.price}\n")


#stick 80 in model, using atm vol
# okay this 80 call is 15 delta


#okay now i know what volatility to use for 15 delta
#use that to price it and then i get price
