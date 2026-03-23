from tools.tool import Tool
import simpleeval

class Calculator(Tool):

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

    def execute(self, args: dict):
        if "expression" not in args:
            raise KeyError('"expression" is missing')
        return simpleeval.simple_eval(args["expression"])