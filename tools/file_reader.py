from tools.tool import Tool
import os

class FileReader(Tool):
    """Tool that reads and returns the full contents of a local text file."""

    def __init__(self):
        name = "file_reader"
        description = "Reads and returns the contents of a local text file. Supports .txt, .csv, .json, .md, .py, and other plain text formats."
        parameters = {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "The path to the file to read, e.g. 'data/report.csv' or 'notes.txt'"
                }
            },
            "required": ["filepath"]
        }
        super().__init__(name, description, parameters)

    def execute(self, args: dict) -> str:
        """Read the file at the given path and return its contents as a string.

        Args:
            args: Must contain a ``"filepath"`` key with the path to the file.

        Returns:
            The full text content of the file.

        Raises:
            KeyError: If ``"filepath"`` is absent from *args*.
            FileNotFoundError: If the file does not exist at the given path.
        """
        if "filepath" not in args:
            raise KeyError('"filepath" is missing')
        
        if not os.path.isfile(args['filepath']):
            raise FileNotFoundError('file does not exist at the given path')

        with open(args['filepath'], "r") as f:
            return f.read()