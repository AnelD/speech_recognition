import asyncio
import json
from typing import Optional

import websockets.asyncio
from websockets import ConnectionClosedOK

from speech_recognition.utils.logger_helper import LoggerHelper

log = LoggerHelper(__name__).get_logger()


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

    async def connect(self) -> None:
        """Connect to the websocket server at given URI."""
        self.ws = await websockets.connect(self.uri)
        asyncio.create_task(self.receive_messages())

    async def send_message(self, message: str) -> None:
        """Send the provided message to the websocket server.

            Calls the message_preprocessor on the message before sending it.
        Args:
            message: The message to send.
        """
        if self.ws:
            log.info(f"Sending message: {message}")
            await self.ws.send(message)
        else:
            log.error("Not connected to the WebSocket server.")

    async def _message_handler(self, message: str) -> None:
        try:
            message = json.loads(message)
            log.info(f"Message received: {message}")
            if message["type"] == "GENERATE_AUDIO_REQUEST":
                text = message["message"]["text"]
                log.info(f"Text to generate audio from: {text}")
                # Check that text isn't empty and doesn't only contain whitespace
                if text.strip() and text:
                    await self.queue.put(text)
                else:
                    await self.send_message(
                        json.dumps(
                            {
                                "type": "GENERATE_AUDIO_ERROR",
                                "message": {
                                    "text": "Text to generate audio for was empty"
                                },
                            }
                        )
                    )
        except Exception as e:
            log.error(f"Exception occurred: {e}")

    async def receive_messages(self) -> None:
        """Handles messages from the websocket server.

        If no handler is provided, messages will only be printed."""
        while True:
            try:
                raw = await self.ws.recv()
                if self._message_handler:
                    await self._message_handler(raw)
                else:
                    log.info(f"Message received: {raw}")
            except ConnectionClosedOK:
                break
            except Exception as e:
                log.error(f"Exception occurred: {e}")
                break

    async def close_connection(self, message: Optional[str] = None) -> None:
        """Close the connection to the websocket server with an optional final message.

        Args:
            message: str: The message to send."""
        if self.ws:
            log.info(f"Closing the connection to the WebSocket server")
            if message:
                log.info(f"Sending final message: {message}")
                await self.ws.send(message)
            await self.ws.close()
            self.ws = None
