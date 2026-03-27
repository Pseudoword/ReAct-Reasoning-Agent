class Memory:
    """Stores the conversation history for a single agent run.

    Holds an ordered list of messages (system, user, assistant, tool)
    that is passed to the LLM on each step of the ReAct loop.
    """

    messages: list[dict]

    def __init__(self):
        self.messages = []

    def add_message(self, message: dict) -> None:
        """Append a copy of *message* to the conversation history.

        Args:
            message: A dict with at least a ``role`` key, e.g.
                ``{"role": "user", "content": "Hello"}``.
        """
        self.messages.append(message.copy())

    def get_message(self) -> list:
        """Return a shallow copy of the full conversation history.

        Returns:
            A list of message dicts in insertion order.
        """
        return self.messages.copy()