from agent.memory import Memory
from tools.tool import Tool
from tools.tool import registry
from openai import OpenAI
import json

def build_tool_schemas(registry: dict):
    tools = []
    for key in registry:
        tool = registry.get(key)
        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
                }
            }
        )
    return tools

def loop(client: OpenAI, user_message: str, system_message: str, max_step: int):
    memory = Memory()
    memory.add_message({"role": "system", "content": system_message})
    memory.add_message({"role": "user", "content": user_message})
    tools = build_tool_schemas(registry)

    for i in range(max_step):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=memory.get_message(),
            tools=tools
        )
        response_message = response.choices[0].message
        memory.add_message(response_message)
        
        if not response_message.tool_calls:
            return response_message.content
        
        for call in response_message.tool_calls:
            function_name = call.function.name
            function_arguments = json.loads(call.function.arguments)
            tool = registry.get(function_name)
            result = tool.execute(function_arguments)
            
            memory.add_message({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result)
                }
            )

    return "Max steps reached"

# function run_agent(question):
#     create memory
#     add system message to memory
#     add user question to memory
#     build tool schemas from registry

#     loop max_steps times:
#         call OpenAI API with memory and tools
#         get the assistant message
#         add assistant message to memory

#         if no tool_calls:
#             return message content    ← final answer

#         for each tool_call:
#             parse the function name and arguments
#             find the tool in registry
#             execute it
#             add tool result to memory

#     return "Max steps reached"        ← safety exit
