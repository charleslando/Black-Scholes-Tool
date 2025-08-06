import os
from dotenv import load_dotenv
import urllib3
import requests

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("ALLASSO_API_KEY")
headers = {"Authorization": f"Key {api_key}"}
def get_allasso_data():


    if not api_key:
        raise ValueError("API key for Allasso is not set. Please set the ALLASSO_API_KEY environment variable.")

    url = "https://allasso.app/data-api/v1/reference/underlying"

    try:
        # Suppress InsecureRequestWarning
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Now you can make the request without the warning
        #response = requests.get(url, verify='/etc/ssl/certs/ca-certificates.crt')
        response = requests.get(url, headers=headers, verify=False)
        #print(response.content)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching data from Allasso: {e}")
        return None


data = get_allasso_data()

print(data)





#
#
# # import requests
# # import os
# #
# #
# # # Setup auth headers
# # # ALLASSO_API_KEY is the environment variable that stores the API key, if it is not set, it will default to api key in the code
# AUTH_HEADERS = {
#     "Authorization": f"Key {os.environ.get('ALLASSO_API_KEY', 'INPUT YOUR API KEY HERE')}"
# }
# # print(AUTH_HEADERS)
# API_URL = "https://allasso.app/data-api/v1"
# #
# #
# def get_reference_underlyings() -> list[dict]:
#     response = requests.get(API_URL + "/v1/reference/underlyings", headers=AUTH_HEADERS)
#     response.raise_for_status()
#     return response.json()
#
#
# get_reference_underlyings()
