import asyncio
import json
import logging
from typing import Optional

import websockets
from websockets import ConnectionClosed

from .logger_helper import LoggerHelper

log = LoggerHelper(__name__, log_level=logging.DEBUG).get_logger()


class WebSocketClient:
    """A client to handle a websocket connection.

    If no handler is provided, messages will only be printed.

    Attributes:
        uri (str): The URI of the websocket server
    """

    def __init__(
        self,
        uri: str,
        queue: asyncio.Queue,
    ) -> None:
        self.uri = uri
        self.queue = queue
        self.ws = None

    async def connect(self):
        """Connect to the websocket server at given URI."""
        self.ws = await websockets.connect(self.uri)
        asyncio.create_task(self.receive_messages())

    # ToDo add logic
    def _message_preprocessor(self, message: str):
        return message

    async def send_message(self, message: str):
        """Send the provided message to the websocket server.

            Calls the message_preprocessor on the message before sending it.
        Args:
            message: The message to send.
        """
        if self.ws:
            if self._message_preprocessor:
                message = self._message_preprocessor(message)
            log.info(f"Sending message: {message}")
            await self.ws.send(message)
        else:
            log.error("Not connected to the WebSocket server.")

    async def _message_handler(self, message: str):
        try:
            message = json.loads(message)
            log.info(f"Message received: {message}")
            if message["type"] == "GENERATE_AUDIO_REQUEST":
                text = message["message"]["text"]
                log.debug("Text to generate audio from: ", text)
                await self.queue.put(text)
        except Exception as e:
            log.error(f"Exception occurred: {e}")

    async def receive_messages(self):
        """Handles messages from the websocket server.

        If no handler is provided, messages will only be printed."""
        while True:
            try:
                raw = await self.ws.recv()
                if self._message_handler:
                    await self._message_handler(raw)
                else:
                    log.info(f"Message received: {raw}")
            except ConnectionClosed as e:
                log.error("Connection to the WebSocket server closed.", e)
                break

    async def close_connection(self, message: Optional[str] = None):
        """Close the connection to the websocket server with an optional final message.

        Args:
            message: str: The message to send."""
        if self.ws:
            log.info(f"Closing the connection to the WebSocket server")
            if message:
                log.info(f"Sending final message: {message}")
                self.ws.send(message)
            await self.ws.close()
            self.ws = None
