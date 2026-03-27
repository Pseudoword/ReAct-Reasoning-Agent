import os
import argparse
from openai import OpenAI
from tools.tool import register_tool
from tools.calculator import Calculator
from tools.web_search import WebSearch
from tools.file_reader import FileReader
from agent.loop import loop

register_tool([Calculator(), WebSearch(), FileReader()])

DEFAULT_QUERY = (
    "What is the population of Canada's capital city divided by "
    "the number of provinces in Canada?"
)


def main():
    parser = argparse.ArgumentParser(description="ReAct Reasoning Agent")
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help="The question to ask the agent",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show the full reasoning trace (thoughts, tool calls, observations)",
    )
    parser.add_argument(
        "--file",
        metavar="PATH",
        help="Path to a local file to include as context",
    )
    args = parser.parse_args()

    system_message = None
    if args.file:
        try:
            with open(args.file, "r") as f:
                file_content = f.read()
            system_message = (
                f"The user has provided the following file ({args.file}):\n\n"
                f"{file_content}"
            )
        except FileNotFoundError:
            print(f"Error: file '{args.file}' not found.")
            return

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    result = loop(client, args.query, system_message=system_message, verbose=args.verbose)
    print(result)


if __name__ == "__main__":
    main()
