from tools.tool import Tool
import simpleeval

class Calculator(Tool):
    """Tool that safely evaluates mathematical expressions using ``simpleeval``."""

    def __init__(self):
        name = "calculator"
        description = "Evaluates a mathematical expression and returns the result. Use for any arithmetic, algebra, or numeric computation."
        parameters = {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g. '1017449 / 10' or '2 ** 8'"
                }
            },
            "required": ["expression"]
        }
        super().__init__(name, description, parameters)

    def execute(self, args: dict) -> int:
        """Evaluate the mathematical expression in *args* and return the result.

        Args:
            args: Must contain an ``"expression"`` key with a string expression.

        Returns:
            The numeric result of the evaluated expression.

        Raises:
            KeyError: If ``"expression"`` is absent from *args*.
        """
        if "expression" not in args:
            raise KeyError('"expression" is missing')
        return simpleeval.simple_eval(args["expression"])