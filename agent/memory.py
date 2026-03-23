class Memory:
    messages: list[dict]

    def __init__(self):
        self.messages = []
    
    def add_message(self, message: dict) -> None:
        self.messages.append(message.copy())

    def get_message(self) -> list:
        return self.messages.copy()