# # A tool meant to exract data from a shorthand trade description
# import pandas as pd
# import tkinter as tk
# from tkinter import simpledialog, messagebox
# from vol_solver import get_days_to_maturity
import pandas as pd
from datetime import datetime

month_codes = {
    'F': 'Jan',
    'G': 'Feb',
    'H': 'Mar',
    'J': 'Apr',
    'K': 'May',
    'M': 'Jun',
    'N': 'Jul',
    'Q': 'Aug',
    'U': 'Sep',
    'V': 'Oct',
    'X': 'Nov',
    'Z': 'Dec'
}

commodity_codes = {
    'CL': 'WTI Crude Oil (Nymex)',
    'CO': 'Brent Crude Oil (ICE)',
    'NG': 'Natural Gas (Nymex)',
    'XB': 'RBOB Gasoline (Nymex)',
    'QS': 'Low Sulfur Gas Oil (ICE Europe)'
}
def get_days_to_maturity(expiry):
    expiry_df = pd.read_csv('Data/Brent_Expiries.csv', index_col=0)
    expiry_date = expiry_df.at[expiry, 'EXPIRY']
    datetime_object = datetime.strptime(expiry_date, '%m/%d/%y')
    delta = datetime_object - datetime.now()
    return delta.days

def parse_structure(structure):
    # commodity = commodity_codes.get(trade[:2], 'Unknown Commodity')
    # month = month_codes.get(trade[2], 'Unknown Month')
    # strike_price = int(trade[3:-1])
    # option_type = 'call' if (trade[-1] == 'c' or trade[-1] == 'C') else ('put' if (trade[-1] == 'p' or trade[-1] == 'P') else 'Unknown Option Type')
    #
    # # Create a pop-up window for confirmation
    # root = tk.Tk()
    # root.withdraw()  # Hide the root window
    #
    # details = f"Commodity: {commodity}\nMonth: {month}\nStrike Price: {strike_price}\nOption Type: {option_type}"
    # response = messagebox.askyesno("Confirm Details", f"Just to clarify, you want:\n\n{details}\n\nIs this correct?")
    #
    # if not response:
    #     # Ask for manual input if the user selects "No"
    #     commodity = simpledialog.askstring("Commodity", f"Commodity ({commodity}):", initialvalue=commodity)
    #     month = simpledialog.askstring("Month", f"Month ({month}):", initialvalue=month)
    #     strike_price = simpledialog.askinteger("Strike Price", f"Strike Price ({strike_price}):", initialvalue=strike_price)
    #     option_type = simpledialog.askstring("Option Type", f"Option Type ({option_type}):", initialvalue=option_type)

    # structure_parts = trade.split()
    # commodity = structure_parts[0]  # First part is the commodity code
    # commodity = commodity_codes.get(commodity, 'Unknown Commodity')
    #
    # month = structure_parts[1]  # Second part is the month code
    # month = month_codes.get(month, 'Unknown Month')
    #
    #
    # strike_price = structure_parts[2]  # Third part is the strike price
    #
    # option_type = "call"
    #
    # # strike_1 =""
    # # strike_2
    #
    #
    # return commodity, month, strike_price, option_type

    try:
        commodity = commodity_codes.get(structure[:2])
        month = month_codes.get(structure[2])
        # Find the longest matching prefix in commodity_codes
        # commodity = None
        # for key in sorted(commodity_codes.keys(), key=len, reverse=True):
        #     if structure.startswith(key):
        #         commodity = commodity_codes[key]
        #         break
        #
        # if not commodity:
        #     raise KeyError(f"Commodity code not found for structure '{structure}'")
        #
        # month = month_codes.get(structure[len(key)], 'Unknown Month')

        #print(month)
        year = structure[-1]
        year = int(year)
        year = year + 20
        year = str(year)

        expiration = month + '-' + year
        # print(expiration)
        T = get_days_to_maturity(expiration)
        days_to_expiration = T
        T = T/365.0

        #print(f"Commodity: {commodity}, Expiration: {expiration}, T: {T}, Days to Expiration: {days_to_expiration}")
        return commodity, T, expiration, days_to_expiration
    except KeyError as e:
        print(f"Error parsing structure '{structure}': {e}")
        return None, None, None


# def calculate_time_to_maturity(month):
#     from datetime import datetime
#     # Assuming the current year is 2023 for this example
#     current_year = datetime.now().year
#     month_number = list(month_codes.values()).index(month) + 1  # Convert month name to number (1-12)
#     maturity_date = datetime(current_year, month_number, 1)  # First day of the month
#     today = datetime.now()
#     time_to_maturity = (maturity_date - today).days / 365.0  # Convert days to years
#     return time_to_maturity


if __name__ == "__main__":
    trade = "CL Z25 75/100 C"



    complex_trade = "CL V25 85.00/105.00 1x2 Call Spread x62.50, 2d"

    print(f"Parsing trade: {trade}")
    #commodity, month, strike_price, option_type = parse_trade(trade)
    #print(f"Final Details:\nCommodity: {commodity}\nMonth: {month}\nStrike Price: {strike_price}\nOption Type: {option_type}")


