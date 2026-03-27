from abc import ABC, abstractmethod
from typing import List

class Tool(ABC):
    """Abstract base class for all agent tools.

    Subclasses must implement :meth:`execute` and call ``super().__init__``
    with the tool's name, description, and JSON-schema parameter definition.
    """

    name: str
    description: str
    parameters: dict

    def __init__(self, name: str, description: str, parameters: dict):
        self.name = name
        self.description = description
        self.parameters = parameters

    def get_name(self) -> str:
        """Return the tool's registered name."""
        return self.name

    def get_description(self) -> str:
        """Return the human-readable description exposed to the LLM."""
        return self.description

    def get_parameters(self) -> dict:
        """Return a copy of the JSON-schema parameter definition."""
        return self.parameters.copy()

    @abstractmethod
    def execute(self, args: dict):
        """Execute the tool with the given arguments.

        Args:
            args: Keyword arguments matching the tool's parameter schema.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError


registry: dict[str, Tool] = {}


def register_tool(tools: List[Tool]) -> None:
    """Add *tool* to the global tool registry, keyed by its name.

    Args:
        tool: A fully initialised :class:`Tool` instance.
    """
    for tool in tools:
        registry[tool.name] = tool