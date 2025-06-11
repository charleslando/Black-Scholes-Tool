import blpapi
import numpy as np
import datetime
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import blackscholes
from blackscholes import Black76Call, Black76Put



# def get_bloomberg_data(ticker, field, start_date, end_date):
#     session = blpapi.Session()
#     if not session.start():
#         raise Exception("Failed to start Bloomberg session.")
#     if not session.openService("//blp/refdata"):
#         raise Exception("Failed to open Bloomberg service.")
#
#     ref_data_service = session.getService("//blp/refdata")
#     request = ref_data_service.createRequest("HistoricalDataRequest")
#     request.append("securities", ticker)
#     request.append("fields", field)
#     request.set("startDate", start_date.strftime("%Y%m%d"))
#     request.set("endDate", end_date.strftime("%Y%m%d"))
#
#     session.sendRequest(request)
#
#     data = []
#     while True:
#         event = session.nextEvent()
#         for msg in event:
#             if msg.hasElement("securityData"):
#                 security_data = msg.getElement("securityData")
#                 for i in range(security_data.numValues()):
#                     data.append(security_data.getValueAsElement(i))
#         if event.eventType() == blpapi.Event.RESPONSE:
#             break
#
#     return data

def plot_grid():


    # Create a grid of data


    # Example Black-Scholes calculations
    F = 100  # Forward price
    K = 100  # Strike price
    T = 1  # Time to expiration in years
    r = 0.0035  # Risk-free interest rate
    sigma = 0.35  # Volatility of the underlying asset

    call_delta, call_gamma, call_price, call_theta, call_vomma, put_delta, put_price = calculate_data(F, K, T, r, sigma)



def calculate_data(F, K, T, r, sigma):

    call = Black76Call(F=F, K=K, T=T, r=r, sigma=sigma)
    call_price = call.price()
    call_delta = call.delta()
    call_vomma = call.vomma()
    call_gamma = call.gamma()
    call_theta = call.theta()
    put = Black76Put(F=F, K=K, T=T, r=r, sigma=sigma)
    put_price = put.price()
    put_delta = put.delta()
    put_vomma = put.vomma()
    put_gamma = put.gamma()
    return call_delta, call_gamma, call_price, call_theta, call_vomma, put_delta, put_price


# Example usage
if __name__ == "__main__":
    plot_grid()