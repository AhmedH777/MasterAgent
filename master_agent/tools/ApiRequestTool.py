import json
import requests
from pydantic import BaseModel, HttpUrl
from typing import Dict, Optional
from master_agent.tools.BaseTool import BaseTool

class ApiRequestTool(BaseTool):
    """Tool for making API requests (GET/POST) using predefined APIs."""

    description = "Sends GET or POST requests to a predefined API and returns the response."

    # Load predefined API mappings
    with open("master_agent/tools/api_mappings.json", "r") as file:
        API_MAPPINGS = json.load(file)

    class InputSchema(BaseModel):
        api_name: str  # Name of the predefined API (e.g., "weather", "bitcoin_price")
        params: Dict[str, str] = {}  # Query parameters (e.g., {"q": "New York", "appid": "your_api_key"})

    def run(self, arguments: dict):
        """Executes the API request based on predefined mappings."""
        args = self.validate_args(arguments)
        api_name, user_params = args.api_name, args.params

        # Check if API exists in the mapping
        if api_name not in self.API_MAPPINGS:
            return json.dumps({"error": f"API '{api_name}' not found in predefined mappings."})

        api_info = self.API_MAPPINGS[api_name]
        url, method, required_params = api_info["url"], api_info["method"].upper(), api_info["params"]

        # Ensure all required params are provided
        missing_params = [p for p in required_params if p not in user_params]
        if missing_params:
            return json.dumps({"error": f"Missing required parameters: {', '.join(missing_params)}"})

        try:
            if method == "GET":
                response = requests.get(url, params=user_params)
            elif method == "POST":
                response = requests.post(url, json=user_params)
            else:
                return json.dumps({"error": "Invalid method. Use GET or POST."})

            return json.dumps(response.json()) if response.ok else json.dumps({"error": response.text})
        except Exception as e:
            return json.dumps({"error": str(e)})
