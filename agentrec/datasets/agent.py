from dataclasses import dataclass
from typing import Optional

@dataclass
class Agent:
    """
    A dataclass which describes an AI agent.
    """
    name: str
    description: Optional[str] = None
    examples: Optional[list[str] | list[dict]] = None
