import json
import pandas as pd
from pydantic import BaseModel
from master_agent.tools.BaseTool import BaseTool
#from BaseTool import BaseTool

class ReadCsvTool(BaseTool):
    """Tool for reading an entire CSV file."""

    description = "Reads only CSV file and returns its entire content as a list of dictionaries."

    class InputSchema(BaseModel):
        file_path: str  # Only requires the file path now

    def run(self, arguments: dict):
        """Reads the entire CSV file and returns JSON as a string."""
        args = self.validate_args(arguments)

        try:
            df = pd.read_csv(args.file_path)
            result = json.dumps(df.to_dict(orient="records"))  # ✅ Convert to JSON string
        except Exception as e:
            result = json.dumps({"error": str(e)})  # ✅ Return errors as JSON string
        
        if result is None:  # ✅ Extra safeguard
            result = json.dumps({"error": "Unexpected null response"})

        return result

'''
# Unit Test
if __name__ == "__main__":
    tool = ReadCsvTool()
    args = {"file_path": "data.csv"}
    print(tool.run(args))  # ➞ '[{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]'
'''