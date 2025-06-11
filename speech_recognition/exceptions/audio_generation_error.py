class AudioGenerationError(Exception):
    """Exception raised for errors during audio generation.

    Attributes:
        message (str): Explanation of the error.
    """

    def __init__(self, message: str) -> None:
        """Initialize AudioGenerationError.

        Args:
            message (str): Error message to describe the exception.
        """
        self.message = message
        super().__init__(self.message)
