from abc import ABC, abstractmethod

class BaseTool(ABC):
    """Abstract base class for tools."""
    
    @abstractmethod
    def run(self, arguments: dict):
        """Execute the tool function with the provided arguments."""
        pass

    @classmethod
    def validate_args(cls, args):
        """Validate input arguments using Pydantic."""
        return cls.InputSchema(**args)

    @classmethod
    def as_dict(cls):
        """Return a JSON-serializable tool definition for LiteLLM."""
        try:
            return {
                "type": "function",
                "function": {
                    "name": cls.__name__.lower(),
                    "description": cls.description,
                    "parameters": cls.InputSchema.model_json_schema()
                }
            }
        except Exception as e:
            print(f"Error generating schema for {cls.__name__}: {e}")
            return {}
