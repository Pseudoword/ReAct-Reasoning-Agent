from tools.tool import Tool
from ddgs import DDGS

class WebSearch(Tool):
    """Tool that queries DuckDuckGo and returns the top text results."""

    def __init__(self):
        name = "web_search"
        description = "Searches the web using DuckDuckGo and returns the top results. Use for looking up facts, current data, or any information not already known."
        parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, e.g. 'population of Ottawa Canada'"
                }
            },
            "required": ["query"]
        }
        super().__init__(name, description, parameters)

    def execute(self, args: dict) -> str:
        """Search the web for *query* and return the top 3 results as plain text.

        Args:
            args: Must contain a ``"query"`` key with the search string.

        Returns:
            A string with each result's title and body separated by newlines.

        Raises:
            KeyError: If ``"query"`` is absent from *args*.
        """
        if "query" not in args:
            raise KeyError('"query" is missing')

        search_results = DDGS().text(args["query"], max_results=3)
        result = ''

        for r in search_results:
            result += f"{r['title']}\n{r['body']}\n\n"

        return result