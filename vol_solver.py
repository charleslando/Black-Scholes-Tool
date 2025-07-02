import numpy as np
import pandas as pd
#from dash_app import calculate_scenario
from datetime import datetime


vol_matrix = pd.read_csv('volatility_matrix.csv', index_col=0)

call_delta_map = {
#    '0.1DP': 0.999,  # Added for completeness, not in original map
#    '1DP': 0.99, # Added for completeness, not in original map

    '10DP': 0.90,
    '15DP': 0.85,
    '25DP': 0.75,
    '35DP': 0.65,
    '50D': 0.5,
    '35DC': 0.35,
    '25DC': 0.25,
    '15DC': 0.15,
    '10DC': 0.10,

#    '1DC': 0.01,    # Added for completeness, not in original map
#    '0.1DC': 0.001,  # Added for completeness, not in original map
}

put_delta_map = {
#    '0.1DP': 0.001,  # Added for completeness, not in original map
#    '1DP': 0.01,     # Added for completeness, not in original map

    '10DP': 0.10,
    '15DP': 0.15,
    '25DP': 0.25,
    '35DP': 0.35,
    '50D': 0.5,
    '35DC': 0.65,
    '25DC': 0.75,
    '15DC': 0.85,
    '10DC': 0.90,

#    '1DC': 0.99,   # Added for completeness, not in original map
#    '0.1DC': .999,  # Added for completeness, not in original map
}

def get_atm_volatility(vol_matrix, expiration):
    try:
        # label‐based lookup—super concise
        #print(expiration)
        return vol_matrix.at[expiration, '50D']
    except KeyError:
        available = vol_matrix.index.tolist()
        raise KeyError(
            f"Could not find expiry {expiration!r} in vol_matrix. "
            f"Available expiries: {available}"
        )

def get_days_to_maturity(expiry):
    expiry_df = pd.read_csv('WTI_Expiries.csv', index_col=0)
    expiry_date = expiry_df.at[expiry, 'EXPIRY']
    datetime_object = datetime.strptime(expiry_date, '%m/%d/%y')
    delta = datetime_object - datetime.now()
    return delta.days


def interpolate_vol(df, expiration, delta, option_type):
    # Insert out of bounds deltas by adding weighted values



    # decay = 0.8
    # start = 10
    # weights = [start * (decay ** i) for i in range(len(df))] # Exclude '1DP' and '1DC'
    #df['Weighting'] = weights

    # ten_dp_vals = df['10DP'].values.tolist()
    # one_dp_vals = [ten_dp + weight for ten_dp, weight in zip(ten_dp_vals, weights)]
    # # Add 1DP column at the beginning
    # vol_matrix.insert(0, '1DP', one_dp_vals)
    # # Add 0.1DP column at the beginning
    # point_one_dp_vals = [one_dp + weight * 2 for one_dp, weight in zip(one_dp_vals, weights)]
    # vol_matrix.insert(0, '0.1DP', point_one_dp_vals)
    #
    # ten_dc_vals = df['10DC'].values.tolist()
    # one_dc_vals = [ten_dc + weight for ten_dc, weight in zip(ten_dc_vals, weights)]
    # # Add 1DC column at the end
    # vol_matrix.insert(len(vol_matrix.columns), '1DC', one_dc_vals)
    # # Add 0.1DC column at the end
    # point_one_dc_vals = [one_dc + weight * 2 for one_dc, weight in zip(one_dc_vals, weights)]
    # vol_matrix.insert(len(vol_matrix.columns), '0.1DC', point_one_dc_vals)

    x_vals = call_delta_map.values() if option_type.lower() == 'call' else put_delta_map.values()
    y_vals = df.loc[expiration].tolist()
    x_vals = list(x_vals)


    # get delta prediction based on x and Y
    poly = np.poly1d(np.polyfit(x_vals, y_vals, deg=3))
    vol_prediction = poly(delta)
    return vol_prediction







    # # Add 1DC column at the end
    # vol_matrix['1DC'] = vol_matrix['10DC'] + weighted_val



    # #Filter relevant data columns
    # if option_type.lower() == 'call':
    #    relevant = {k: v for k, v in call_delta_map.items()}
    # else:
    #    relevant = {k: v for k, v in put_delta_map.items()}




    # delta_vals = list(relevant.values())
    # columns = list(relevant.keys())

    # weight_delta = 1
    # weight = df.at[month, 'Weighted']
    # if delta < min(delta_vals) or delta > max(delta_vals):
    #     if(delta < min(delta_vals)):
    #         #interpolate between 10 and weight_delta
    #         delta_vals.insert(delta)



    #
    # #find the two bracketing deltas
    # for i in range(len(delta_vals) - 1):
    #     d1, d2 = delta_vals[i], delta_vals[i + 1]
    #     if d1 <= delta <= d2 or d2 <= delta <= d1:
    #         # Interpolate between the two deltas
    #         col1, col2 = columns[i], columns[i + 1]
    #         vol1 = df.at[month, col1]
    #         vol2 = df.at[month, col2]
    #
    #         # Linear interpolation
    #         weight = (delta - d1) / (d2 - d1)
    #         interpolated_vol = vol1 + weight * (vol2 - vol1)
    #         return interpolated_vol




#    raise ValueError(f"Delta {delta} not found in the specified month {month} for option type {option_type}. Perhaps the delta is out of bounds?")
# def interpolate_vol(df, month, delta, option_type):
#     import numpy as np
#
#     # Filter relevant data columns based on option type
#     if option_type.lower() == 'call':
#         delta_map = call_delta_map
#     else:
#         delta_map = put_delta_map
#
#     # Ensure data is ordered by delta values
#     deltas = list(delta_map.values())
#     cols = list(delta_map.keys())
#     sorted_pairs = sorted(zip(deltas, cols))
#     sorted_deltas, sorted_cols = zip(*sorted_pairs)
#
#     # Get existing vols for the month
#     try:
#         y_vals = [df.at[month, col] for col in sorted_cols]
#     except KeyError:
#         raise ValueError(f"Month {month} or one of the columns {sorted_cols} not found.")
#
#     x_vals = np.array(sorted_deltas)
#     y_vals = np.array(y_vals)
#
#     # Fit polynomial curve
#     coeffs = np.polyfit(x_vals, y_vals, deg=3)  # You can try deg=2 or 4 for tuning
#     poly = np.poly1d(coeffs)
#
#     # Predict missing vols and inject into DataFrame
#     for label, x in [('0.1DP', 0.999), ('1DP', 0.99), ('1DC', 0.01), ('0.1DC', 0.001)]:
#         if label not in df.columns:
#             df[label] = np.nan  # Create column if missing
#         df.at[month, label] = poly(x)
#
#     # Interpolate for requested delta
#     return float(poly(delta))




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





























"""
# constants for the Black-Scholes model


expiry = 'Jul-26'

F = 100
K = 160
T = get_days_to_maturity(expiry) / 365.0  # Convert days to years
r = 0.05
option_type = 'call'
month = expiry[:-3]
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
"""