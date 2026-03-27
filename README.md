# ReAct Reasoning Agent

A Python-based autonomous agent that solves multi-step questions by iteratively reasoning about what to do, selecting and executing tools, observing results, and refining its approach — following the [ReAct](https://arxiv.org/abs/2210.03629) (Reasoning + Acting) framework.

## Motivation

Most LLM applications treat the model as a single-shot question-answerer. Real-world problems often require multiple steps: looking something up, performing a calculation on the result, and then synthesizing an answer. This project implements the core agent loop that underpins modern agentic AI systems — the same paradigm used in scientific agents, coding assistants, and autonomous research tools.

## How It Works

```
User Question
     │
     ▼
┌──────────────┐
│  LLM Reasons │◄──────────────────┐
│  about next  │                   │
│  action      │                   │
└──────┬───────┘                   │
       │                           │
       ▼                           │
  ┌─────────┐    Yes    ┌─────────────────┐
  │ Tool     │────────►  │ Execute tool,   │
  │ called?  │           │ return result   │
  └────┬─────┘           └─────────────────┘
       │ No
       ▼
┌──────────────┐
│ Final Answer │
└──────────────┘
```

At each step, the agent:
1. Receives the full conversation history (question + all prior reasoning and tool results).
2. Decides whether to call a tool or deliver a final answer.
3. If a tool is called, the result is appended to the conversation, and the loop continues.
4. The loop terminates when the agent produces a final text response or hits the max-step limit.

## Tools

| Tool | Description | Example Use |
|------|-------------|-------------|
| **Web Search** | Queries DuckDuckGo and returns top results | "What is the population of Ottawa?" |
| **Calculator** | Evaluates mathematical expressions safely | "What is 1039301 / 3?" |
| **File Reader** | Reads and returns contents of a local text or CSV file | "Summarize the data in report.csv" |

Tools are defined with a name, description, and parameter schema — matching OpenAI's native function-calling interface. New tools can be added by implementing a single Python function and registering its schema.

## Example

**Query:** "What is the population of Canada's capital city divided by the number of provinces in Canada?"

**Agent trajectory:**
```
Step 1 [Reason]:  I need to find the population of Canada's capital. Let me search for it.
Step 1 [Action]:  web_search("population of Ottawa Canada")
Step 1 [Observe]: Ottawa has a population of approximately 1,017,449.

Step 2 [Reason]:  Now I need the number of provinces in Canada.
Step 2 [Action]:  web_search("number of provinces in Canada")
Step 2 [Observe]: Canada has 10 provinces.

Step 3 [Reason]:  I can now calculate the result.
Step 3 [Action]:  calculator("1017449 / 10")
Step 3 [Observe]: 101744.9

Step 4 [Answer]:  The population of Ottawa (approximately 1,017,449) divided by
                  the number of Canadian provinces (10) is about 101,745.
```

## Project Structure

```
ReAct-Reasoning-Agent/
├── agent/
│   ├── loop.py          # Core ReAct agent loop
│   ├── memory.py        # Conversation history management
│   └── config.py        # Max steps and API settings
├── tools/
│   ├── tool.py          # Tool base class and registry
│   ├── web_search.py    # DuckDuckGo search wrapper
│   ├── calculator.py    # Safe math expression evaluator
│   └── file_reader.py   # Local file reader
├── data/
│   ├── countries.txt    # Sample country list for demos
│   └── sample.csv       # Sample numeric data for demos
├── examples/
│   ├── demo_queries.py  # 9 demo scenarios with full trace output
│   └── trajectories.log # Pre-recorded example trajectories
├── main.py              # CLI entry point
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/Pseudoword/ReAct-Reasoning-Agent.git
cd ReAct-Reasoning-Agent
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
```

## Usage

```bash
# Ask a question
python main.py "What is the GDP per capita of the country that won the 2024 Olympics medal count?"

# Verbose mode (shows full reasoning trace)
python main.py --verbose "Compare the populations of Toronto and Vancouver."

# Use a local file as context
python main.py --file data.csv "What is the average value in the second column?"
```

## Technical Details

- **LLM:** GPT-4o (via OpenAI API) with native function-calling support
- **Agent pattern:** ReAct — interleaved reasoning and action with full conversation memory
- **Tool dispatch:** Structured function calls parsed from the model's response; results appended as `tool` role messages
- **Safety:** Calculator uses `simpleeval` instead of `eval()` to prevent arbitrary code execution; agent loop capped at configurable max iterations (default: 5)
- **Memory:** Full message history maintained across steps, providing the agent with complete episodic context for multi-hop reasoning

## Roadmap

- [x] Project structure and tool interface design
- [x] README and architecture documentation
- [x] Core agent loop with OpenAI function calling
- [x] Web search tool (DuckDuckGo)
- [x] Calculator tool (simpleeval)
- [x] File reader tool
- [x] Error handling and max-iteration safeguards
- [x] Demo examples with logged trajectories
- [x] Retry logic for failed tool calls
- [x] Token usage tracking and cost logging

## Acknowledgments

Inspired by the [ReAct paper](https://arxiv.org/abs/2210.03629) (Yao et al., 2022).

## License

MIT
