from agent.memory import Memory
from tools.tool import Tool
from tools.tool import registry
import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionToolParam
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import json

def _build_tool_schemas(registry: dict[str, Tool]) -> list[ChatCompletionToolParam]:
    """Convert the tool registry into the JSON schema format expected by the OpenAI API.

    Args:
        registry: Mapping of tool name to :class:`~tools.tool.Tool` instance.

    Returns:
        A list of ``{"type": "function", "function": {...}}`` dicts ready to
        pass as the ``tools`` parameter of a chat completion request.
    """
    tools: list[ChatCompletionToolParam] = []
    for key in registry:
        tool = registry[key]
        tools.append({
            "type": "function",
            "function": {
                "name": tool.get_name(),
                "description": tool.get_description(),
                "parameters": tool.get_parameters()
                }
            }
        )
    return tools

def _tool_call(memory: Memory, call: ChatCompletionMessageToolCall) -> None:
    """Execute a single tool call and append the result to memory."""
    function_name = call.function.name
    function_arguments = json.loads(call.function.arguments)

    if function_name not in registry:
        memory.add_message({
            "role": "tool",
            "tool_call_id": call.id,
            "content": f"Error: unknown tool '{function_name}'"
        })
        return

    tool = registry[function_name]
    try:
        result = tool.execute(function_arguments)
    except Exception as e:
        result = f"Error: {e}"

    memory.add_message({
        "role": "tool",
        "tool_call_id": call.id,
        "content": str(result)
    })


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)),
    reraise=True,
)
def _call_api(client: OpenAI, messages, tools):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )


def _print_usage(prompt_tokens: int, completion_tokens: int) -> None:
    print(f"\n[usage] prompt={prompt_tokens}  completion={completion_tokens}  total={prompt_tokens + completion_tokens}")


def loop(client: OpenAI, user_message: str, system_message: str | None = None, max_step: int = 5):
    """Run the ReAct agent loop until the model produces a final answer or steps are exhausted.

    On each iteration the model is called with the full conversation history.
    If the model requests tool calls they are dispatched, their results appended
    to memory, and the loop continues.  When the model replies without tool
    calls its text content is returned as the final answer.

    Args:
        client: An initialised :class:`openai.OpenAI` client.
        user_message: The user's initial question or instruction.
        system_message: The system prompt that defines the agent's behaviour.
        max_step: Maximum number of LLM calls before giving up.

    Returns:
        The model's final text response, or ``"Max steps reached"`` if the
        loop runs out of steps without a conclusive answer.
    """
    memory = Memory()
    if system_message is not None:
        memory.add_message({"role": "system", "content": system_message})
    memory.add_message({"role": "user", "content": user_message})
    tools = _build_tool_schemas(registry)

    prompt_tokens = 0
    completion_tokens = 0

    for _ in range(max_step):
        response = _call_api(client, memory.get_message(), tools)
        if response.usage:
            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens

        response_message = response.choices[0].message
        memory.add_message({k: v for k, v in response_message.model_dump().items() if v is not None})

        if not response_message.tool_calls:
            _print_usage(prompt_tokens, completion_tokens)
            return response_message.content

        for call in response_message.tool_calls:
            if isinstance(call, ChatCompletionMessageToolCall):
                _tool_call(memory, call)

    _print_usage(prompt_tokens, completion_tokens)
    return "Max steps reached"