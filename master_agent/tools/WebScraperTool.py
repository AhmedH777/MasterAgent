import json
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
from master_agent.tools.BaseTool import BaseTool

class WebScraperTool(BaseTool):
    """Tool for scraping web pages and extracting text content."""

    description = "Reads a webpage and returns the main text content."

    class InputSchema(BaseModel):
        url: str  # The webpage URL

    def run(self, arguments: dict):
        """Fetches and scrapes content from the given webpage."""
        args = self.validate_args(arguments)
        url = args.url

        headers = {"User-Agent": "Mozilla/5.0"}  # Prevent getting blocked
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                return json.dumps({"error": f"Failed to fetch page. Status code: {response.status_code}"})
            
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            content = " ".join(p.text for p in paragraphs)

            return json.dumps({"url": url, "content": content[:1000]})  # Limit output length
        except Exception as e:
            return json.dumps({"error": str(e)})
