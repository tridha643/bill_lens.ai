"""
    CDG Client - An example client for the Congress.gov API.

    @copyright: 2022, Library of Congress
    @license: CC0 1.0
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class CDGClient:
    """ A sample client to interface with Congress.gov. """

    def __init__(
        self,
        api_key,
        api_version="v3",
        response_format="json",
        raise_on_error=True,
    ):
        self.api_key = api_key  # Add this line to set the api_key attribute
        self.response_format = response_format  # Add this line to set the response_format attribute
        self.base_url = f"https://api.congress.gov/{api_version}/"
        self._session = requests.Session()

        # Set up retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        # do not use url parameters, even if offered, use headers
        self._session.params = {"format": response_format}
        self._session.headers.update({"x-api-key": api_key})

        if raise_on_error:
            self._session.hooks = {
                "response": lambda r, *args, **kwargs: r.raise_for_status()
            }

    def get(self, endpoint, timeout=30):
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": f"application/{self.response_format}"
        }
        try:
            response = self._session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            if response.headers.get("content-type", "").startswith("application/json"):
                return response.json(), response
            elif response.headers.get("content-type", "").startswith("application/xml"):
                return response.text, response
            else:
                return None, response
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None, None