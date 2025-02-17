import io
import json
import sys
from pydantic import BaseModel
from contextlib import redirect_stdout
from master_agent.tools.BaseTool import BaseTool

class PythonExecutionTool(BaseTool):
    """Tool for executing Python code securely."""

    description = "Executes Python code in a restricted environment."

    class InputSchema(BaseModel):
        code: str  # Python code snippet to execute

    def run(self, arguments: dict):
        """Executes Python code and returns output."""
        args = self.validate_args(arguments)
        code = args.code

        # Restricted built-in functions (Prevent security risks)
        safe_builtins = {
            "print": print,
            "range": range,
            "len": len,
            "sum": sum,
            "abs": abs,
            "min": min,
            "max": max,
            "round": round,
            "sorted": sorted,
            "map": map,
            "filter": filter,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool
        }

        restricted_globals = {
            "__builtins__": safe_builtins
        }

        stdout_capture = io.StringIO()
        try:
            with redirect_stdout(stdout_capture):
                exec(code, restricted_globals, {})
            output = stdout_capture.getvalue()
        except Exception as e:
            output = f"Error: {str(e)}"

        return json.dumps({"output": output})
