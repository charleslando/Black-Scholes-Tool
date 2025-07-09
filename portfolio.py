class Portfolio:

    def __init__(self, price, delta, gamma, theta, vega, premium):
        self.price = price
        self.delta = delta
        self.gamma = gamma
        self.theta = theta
        self.vega = vega
        self.premium = premium
        self.p_and_l = 0
        self.premium = premium
        # self.quantity = quantity


    def to_dict(self):
        return {
            'Average Price': self.price,
            'Delta': self.delta,
            'Gamma': self.gamma,
            'Theta': self.theta,
            'Vega': self.vega,
            'P&L': self.p_and_l,
            'Total Premium': self.premium,
            # 'quantity': self.quantity
        }

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([self.to_dict()])

    def to_json(self):
        import json
        return json.dumps(self.to_dict())

    def to_plotly_format(self):
        #return format_plot_data(self.Premium, self.delta, self.gamma, self.theta, self.vega, self.p_and_l)
        return f"Average Price: {self.price:.2f}<br>Delta: {self.delta:.4f}<br>Gamma: {self.gamma:.2f}<br>Vega: {self.vega:.2f}<br>Theta: {self.theta:.2f}<br>Total Premium: {self.premium:.2f}<br>P&L: {self.p_and_l:.2f}"
        # for variable in printing_variables:
        #     if variable not in self.to_dict():
        #         raise ValueError(f"Variable '{variable}' not found in Portfolio object.")
        # return {

    def calculate_p_and_l(self, original_premium):
        self.p_and_l = self.premium - original_premium
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

    def __str__(self):
        return f"Portfolio: Average Price={self.price}, Delta={self.delta}, Gamma={self.gamma}, Theta={self.theta}, Vega={self.vega}, P&L={self.p_and_l}"

