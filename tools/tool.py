class Tool:
    name: str
    description: str
    parameters: dict

    def __init__(self, name, description, parameters):
        self.name = name
        self.description = description
        self.parameters = parameters
    
    def execute(self, args: dict):
        raise NotImplementedError

registry = {}

def register_tool(tool: Tool):
    registry[tool.name] = tool