import os
from io import BytesIO

import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("ALLASSO_API_KEY")
headers = {"Authorization": f"Key {api_key}"}
def get_allasso_data():


    if not api_key:
        raise ValueError("API key for Allasso is not set. Please set the ALLASSO_API_KEY environment variable.")

    url = "'https://allasso.app/data-api/v1"


    params = {

    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching data from Allasso: {e}")
        return None


data = get_allasso_data()


def call_api_ohlc(ticker: str, headers: dict = headers) -> dict:
    """
    Function that sends the request to the Allasso API to obtain data
    """
    params = {"underlying_id": ticker, "format": "json"}
    response = requests.get(
        "https://allasso.app/api_int/v1/equity/full-history",
        params=params,
        headers=headers,
    )
    response.raise_for_status()

    return pd.read_parquet(BytesIO(response.content))

print(call_api_ohlc("BRN"))  # Example ticker for Brent Crude Oil








