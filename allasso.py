# import os
# from io import BytesIO
#
# import pandas as pd
# import requests
# from dotenv import load_dotenv
#
# # Load environment variables from .env file
# load_dotenv()
# api_key = os.getenv("ALLASSO_API_KEY")
# headers = {"Authorization": f"Key {api_key}"}
# def get_allasso_data():
#
#
#     if not api_key:
#         raise ValueError("API key for Allasso is not set. Please set the ALLASSO_API_KEY environment variable.")
#
#     url = "'https://allasso.app/data-api/v1"
#
#
#     params = {
#
#     }
#
#     try:
#         response = requests.get(url, headers=headers, params=params)
#         response.raise_for_status()  # Raise an error for bad responses
#         data = response.json()
#         return data
#     except requests.exceptions.RequestException as e:
#         print(f"An error occurred while fetching data from Allasso: {e}")
#         return None
#
#
# data = get_allasso_data()
#
#
# def call_api_ohlc(ticker: str, headers: dict = headers) -> dict:
#     """
#     Function that sends the request to the Allasso API to obtain data
#     """
#     params = {"underlying_id": ticker, "format": "json"}
#     response = requests.get(
#         "https://allasso.app/api_int/v1/equity/full-history",
#         params=params,
#         headers=headers,
#     )
#     response.raise_for_status()
#
#     return pd.read_parquet(BytesIO(response.content))
#
# print(call_api_ohlc("BRN"))  # Example ticker for Brent Crude Oil


import requests
import os

# Setup auth headers
# ALLASSO_API_KEY is the environment variable that stores the API key, if it is not set, it will default to api key in the code
AUTH_HEADERS = {
    "Authorization": f"Key {os.environ.get('ALLASSO_API_KEY', 'INPUT YOUR API KEY HERE')}"
}
API_URL = "https://allasso.app/content/1fa115c4-38e9-4cfa-a4ad-3c8578e26459"


def get_reference_underlyings() -> list[dict]:
    response = requests.get(API_URL + "/v1/reference/underlyings", headers=AUTH_HEADERS)
    response.raise_for_status()
    return response.json()


get_reference_underlyings()
