#from dash_app import format_plot_data
class Portfolio:
    def __init__(self, price, delta, gamma, theta, vega, quantity):
        self.price = price
        self.delta = delta
        self.gamma = gamma
        self.theta = theta
        self.vega = vega
        self.p_and_l = 0
        self.quantity = quantity

    def __repr__(self):
        return f"Portfolio(price={self.price}, delta={self.delta}, gamma={self.gamma}, theta={self.theta}, vega={self.vega}, p_and_l={self.p_and_l}, quantity={self.quantity})"

    def to_dict(self):
        return {
            'price': self.price,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'p_and_l': self.p_and_l,
            'quantity': self.quantity
        }

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([self.to_dict()])

    def to_json(self):
        import json
        return json.dumps(self.to_dict())

    def to_plotly_format(self):
        #return format_plot_data(self.price, self.delta, self.gamma, self.theta, self.vega, self.p_and_l)
        return f"Price: {self.price:.2f}<br>Delta: {self.delta:.2f}<br>Gamma: {self.gamma:.2f}<br>Theta: {self.theta:.2f}<br>Vega: {self.vega:.2f}<br>P&L: {self.p_and_l:.2f}<br>Quantity: {self.quantity}"

    def calculate_p_and_l(self, original_price):
        self.p_and_l = self.price - original_price
        return self.p_and_l

    def get_attribute(self, attr):
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            raise AttributeError(f"Portfolio has no attribute '{attr}'")

    def set_attribute(self, attr, value):
        if hasattr(self, attr):
            setattr(self, attr, value)
        else:
            raise AttributeError(f"Portfolio has no attribute '{attr}'")

    def __getitem__(self, item):
        if hasattr(self, item):
            return getattr(self, item)
        else:
            raise KeyError(f"Portfolio has no attribute '{item}'")

    def __setitem__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Portfolio has no attribute '{key}'")

    def __len__(self):
        return len(self.to_dict())

    def __iter__(self):
        for key in self.to_dict():
            yield key, self.to_dict()[key]

    def __contains__(self, item):
        return item in self.to_dict()

    def __eq__(self, other):
        if isinstance(other, Portfolio):
            return self.to_dict() == other.to_dict()
        return False

    def __ne__(self, other):
        if isinstance(other, Portfolio):
            return self.to_dict() != other.to_dict()
        return True
