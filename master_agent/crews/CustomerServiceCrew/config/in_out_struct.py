from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Set, Tuple

class UserRequirements(BaseModel):
    programming_language: str = Field(..., title="Programming language")
    requirements: str = Field(..., title="User requirements")

