import os
from openai import OpenAI
from tools.tool import register_tool
from tools.calculator import Calculator
from tools.web_search import WebSearch
from tools.file_reader import FileReader
from agent.loop import loop

register_tool([Calculator(), WebSearch(), FileReader()])

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    client = OpenAI(api_key=OPENAI_API_KEY)
    result = loop(client, "What is the population of Canada's capital city divided by the number of provinces in Canada?")
    print(result)

if __name__ == "__main__":
    main()