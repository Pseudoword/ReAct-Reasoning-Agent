import os
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a professional storyteller."},
            {"role": "user", "content": "Write a one-sentence bedtime story about a unicorn."}
        ]
    )

    print(response.choices[0].message)

if __name__ == "__main__":
    main()