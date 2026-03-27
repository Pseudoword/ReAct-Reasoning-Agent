"""Configuration constants for the ReAct agent.

Reads secrets from environment variables and defines global limits
used by the agent loop.
"""

import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_STEP = 10