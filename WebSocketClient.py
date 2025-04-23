import asyncio
from typing import Callable, Optional

import websockets
from websockets import ConnectionClosed


class WebSocketClient:
    """A client to handle a websocket connection.

    If no handler is provided, messages will only be printed.

    Attributes:
        uri (str): The URI of the websocket server
        message_handler (Optional[Callable[[str], None]] = None): Handler for incoming messages.
        message_preprocessor (Optional[Callable[[str], str]] = None): Preprocessor for outgoing messages.
    """

    def __init__(
        self,
        uri: str,
        message_handler: Optional[Callable[[str], None]],
        message_preprocessor: Optional[Callable[[str], str]],
    ) -> None:
        self.uri = uri
        self.ws = None
        self.message_handler = message_handler
        self.message_preprocessor = message_preprocessor
        print(
            f"Initialized WebSocketClient with uri: {uri} and message_handler: {message_handler.__name__}"
        )

    async def connect(self):
        """Connect to the websocket server at given URI."""
        self.ws = await websockets.connect(self.uri)
        asyncio.create_task(self.receive_messages())

    async def send_message(self, message: str):
        """Send the provided message to the websocket server.

            Calls the message_preprocessor on the message before sending it.
        Args:
            message: str: The message to send.
        """
        if self.ws:
            if self.message_preprocessor:
                self.message_preprocessor(message)
            print(f"Sending message: {message}")
            await self.ws.send(message)
        else:
            print("Not connected to the WebSocket server.")

    # ToDo: Delete this
    # async def consume(self, raw):
    #     try:
    #         print(f"Raw message: {raw}")
    #         # message = Message.model_validate_json(raw)
    #         message = json.loads(raw)
    #         print(f"Message received: {message}")
    #         if message["type"] == "GENERATE_AUDIO_REQUEST":
    #             text = message["message"]["text"]
    #             print("Text to generate audio from: ", message)
    #             await self.queue.put(text)
    #     except Exception as e:
    #         print(f"Exception occurred: {e}")

    async def receive_messages(self):
        """Handles messages from the websocket server.

        If no handler is provided, messages will only be printed."""
        while True:
            try:
                raw = await self.ws.recv()
                if self.message_handler:
                    self.message_handler(raw)
                else:
                    print(f"Message received: {raw}")
            except ConnectionClosed:
                print("Connection to the WebSocket server closed.")
                break

    async def close_connection(self, message: Optional[str] = None):
        """Close the connection to the websocket server with an optional final message.

        Args:
            message: str: The message to send."""
        if self.ws:
            print(f"Closing the connection to the WebSocket server")
            if message:
                print(f"Sending final message: {message}")
                self.ws.send(message)
            await self.ws.close()
            self.ws = None
