class TranscriptionError(Exception):
    """Exception raised for transcription errors.

    Attributes:
        message (str): Explanation of the error.
    """

    def __init__(self, message: str) -> None:
        """Initialize TranscriptionError.

        Args:
            message (str): Error message to describe the exception.
        """
        self.message = message
        super().__init__(self.message)
