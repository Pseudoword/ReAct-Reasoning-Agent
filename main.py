import os
from openai import OpenAI
from tools.tool import register_tool
from tools.calculator import Calculator
from agent.loop import loop

register_tool(Calculator())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    client = OpenAI(api_key=OPENAI_API_KEY)
    result = loop(client, "What is (245 * 18) + (372 / 12)?", "You are a research agent that solves problems step by step.", 7)
    print(result)

    # response = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[
    #         {"role": "system", "content": "You are a professional storyteller."},
    #         {"role": "user", "content": "Write a one-sentence bedtime story about a unicorn."}
    #     ]
    # )

    # print(response.choices[0].message)

if __name__ == "__main__":
    main()