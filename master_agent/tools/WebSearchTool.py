import json
import requests
from pydantic import BaseModel
from master_agent.tools.BaseTool import BaseTool

class WebSearchTool(BaseTool):
    """Tool for performing web searches using Google Custom Search API."""

    description = "Performs a web search and returns the top results."

    class InputSchema(BaseModel):
        query: str  # The search query

    def run(self, arguments: dict):
        """Executes the web search and returns results."""
        args = self.validate_args(arguments)
        api_key = "AIzaSyC8NwiErvPrQDP3HCrSY7tMIYRtUHXldc8"  # 🔄 Replace with actual API key
        cse_id = "9055bb2566d8c4cd2"  # 🔄 Replace with actual CSE ID

        search_url = f"https://www.googleapis.com/customsearch/v1?q={args.query}&key={api_key}&cx={cse_id}"

        try:
            response = requests.get(search_url)
            data = response.json()

            results = []
            for item in data.get("items", []):
                results.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                })

            return json.dumps(results) if results else json.dumps({"message": "No results found."})
        except Exception as e:
            return json.dumps({"error": str(e)})
