"""
demo_queries.py — Showcase the ReAct agent across different tool combinations.

Run from the project root:
    python examples/demo_queries.py

Each query prints the full reasoning trajectory:
    Step N
      Thought   : <model reasoning>
      Tool      : <tool name>
      Args      : <json arguments>
      Obs       : <tool result>
    --> FINAL ANSWER
"""

import os
import sys
import json
import textwrap

# Allow running from examples/ or project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCall

from agent.memory import Memory
from agent.loop import _build_tool_schemas, _call_api, _tool_call
from tools.tool import registry, register_tool
from tools.calculator import Calculator
from tools.web_search import WebSearch
from tools.file_reader import FileReader

register_tool([Calculator(), WebSearch(), FileReader()])

SYSTEM_PROMPT = (
    "You are a helpful reasoning agent. Think step by step. "
    "Use tools to look up information or perform calculations. "
    "Give a clear, concise final answer once you have enough information."
)

WIDTH = 72
SEP = "=" * WIDTH
INDENT = " " * 13  # aligns continuation lines under the label value


def _wrap(text: str) -> str:
    return textwrap.fill(text.strip(), width=WIDTH - 13, subsequent_indent=INDENT)


def run_demo(client: OpenAI, title: str, query: str, max_steps: int = 8) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(f"  Q: {query}")
    print(SEP)

    memory = Memory()
    memory.add_message({"role": "system", "content": SYSTEM_PROMPT})
    memory.add_message({"role": "user", "content": query})
    tools = _build_tool_schemas(registry)

    prompt_tokens = 0
    completion_tokens = 0

    for step in range(1, max_steps + 1):
        response = _call_api(client, memory.get_message(), tools)
        if response.usage:
            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens

        msg = response.choices[0].message
        memory.add_message({k: v for k, v in msg.model_dump().items() if v is not None})

        print(f"\n  Step {step}")

        if msg.content:
            print(f"    Thought  : {_wrap(msg.content)}")

        if not msg.tool_calls:
            print(f"\n  --> FINAL ANSWER")
            break

        for call in msg.tool_calls:
            if not isinstance(call, ChatCompletionMessageToolCall):
                continue

            tool_name = call.function.name
            tool_args = json.loads(call.function.arguments)
            print(f"    Tool     : {tool_name}")
            print(f"    Args     : {json.dumps(tool_args)}")

            _tool_call(memory, call)

            obs = memory.get_message()[-1].get("content", "")
            if len(obs) > 500:
                obs = obs[:500] + "... [truncated]"
            print(f"    Obs      : {_wrap(obs)}")
    else:
        print("\n  --> MAX STEPS REACHED")

    print(
        f"\n  [tokens] prompt={prompt_tokens}  completion={completion_tokens}"
        f"  total={prompt_tokens + completion_tokens}"
    )


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # 1. Single tool — calculator
    # ------------------------------------------------------------------
    run_demo(
        client,
        "1. Single tool — calculator",
        "What is 2 to the power of 32?",
    )

    # ------------------------------------------------------------------
    # 2. Single tool — web search
    # ------------------------------------------------------------------
    run_demo(
        client,
        "2. Single tool — web search",
        "Who is the current Prime Minister of Japan?",
    )

    # ------------------------------------------------------------------
    # 3. Multi-hop — search + calculator (capital population / provinces)
    # ------------------------------------------------------------------
    run_demo(
        client,
        "3. Multi-hop — search + calculator (capital pop ÷ provinces)",
        "What is the population of Canada's capital city divided by "
        "the number of provinces in Canada?",
    )

    # ------------------------------------------------------------------
    # 4. Multi-hop — search + calculator (GDP / population)
    # ------------------------------------------------------------------
    run_demo(
        client,
        "4. Multi-hop — search + calculator (GDP ÷ population)",
        "What is the GDP of France divided by the population of Germany?",
    )

    # ------------------------------------------------------------------
    # 5. Multi-hop — search + search (city comparison)
    # ------------------------------------------------------------------
    run_demo(
        client,
        "5. Multi-hop — search + search (city populations)",
        "Compare the populations of Toronto and Vancouver.",
    )

    # ------------------------------------------------------------------
    # 6. File + calculator (average of CSV values)
    # ------------------------------------------------------------------
    run_demo(
        client,
        "6. File + calculator (CSV average)",
        "What is the average of the values in data/sample.csv?",
    )

    # ------------------------------------------------------------------
    # 7. Three tools chained — file + search + calculator
    # ------------------------------------------------------------------
    run_demo(
        client,
        "7. Three tools chained — file + search + calculator",
        "Read the countries in data/countries.txt, look up the population "
        "of the first country listed, and divide that number by 1000.",
    )

    # ------------------------------------------------------------------
    # 8. Edge case — nonexistent file (should recover gracefully)
    # ------------------------------------------------------------------
    run_demo(
        client,
        "8. Edge case — missing file (graceful error handling)",
        "Read the file nonexistent.txt and tell me its contents.",
    )

    # ------------------------------------------------------------------
    # 9. Edge case — division by zero (should return the error, not crash)
    # ------------------------------------------------------------------
    run_demo(
        client,
        "9. Edge case — division by zero (graceful error handling)",
        "What is 1 / 0?",
    )


if __name__ == "__main__":
    main()
