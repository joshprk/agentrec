from dataclasses import dataclass
from typing import Optional

@dataclass
class Agent:
    """
    A dataclass which describes an AI agent.
    """
    name: str
    description: Optional[str] = None
    examples: list[str | dict] = []

    @staticmethod
    def from_jsonl(data: dict):
        """
        Returns an `Agent` that describes the given jsonlines serialization.
        """
        return Agent(name=data["name"],
                     description=data["description"],
                     examples=data["examples"])

    def to_jsonl(self):
        """
        Returns a serialization of the AI agent in jsonlines form.
        """
        return {
            "name": self.name,
            "description": self.description,
            "examples": self.examples,
        }
