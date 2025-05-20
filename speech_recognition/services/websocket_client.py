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
        __uri (str): The URI of the websocket server
    """

    def __init__(self, uri: str, queue: asyncio.Queue) -> None:
        self.__uri = uri
        self.__queue = queue
        self.__ws = None

    async def connect(self) -> None:
        """Connect to the websocket server at given URI."""
        self.__ws = await websockets.connect(self.__uri)
        asyncio.create_task(self._receive_messages())

    async def close_connection(self, message: Optional[str] = None) -> None:
        """Close the connection to the websocket server with an optional final message.

        Args:
            message: str: The message to send."""
        if self.__ws:
            log.info(f"Closing the connection to the WebSocket server")
            if message:
                try:
                    log.info(f"Sending final message: {message}")
                    await self.__ws.send(message)
                except ConnectionClosedOK:
                    log.info(
                        "Server already closed the connection couldn't send final message"
                    )
            await self.__ws.close()
            self.__ws = None

    async def send_message(self, message: str) -> None:
        """Send the provided message to the websocket server.

            Calls the message_preprocessor on the message before sending it.
        Args:
            message: The message to send.
        """
        if self.__ws:
            try:
                log.info(f"Sending message: {message}")
                await self.__ws.send(message)
            except Exception as e:
                raise e
        else:
            log.error("Not connected to the WebSocket server.")

    async def _receive_messages(self) -> None:
        """Handles messages from the websocket server.

        If no handler is provided, messages will only be printed."""
        while True:
            try:
                raw = await self.__ws.recv()
                if self.__message_handler:
                    await self.__message_handler(raw)
                else:
                    log.info(f"Message received: {raw}")
            except ConnectionClosedOK:
                break
            except Exception as e:
                log.error(f"Exception occurred: {e}")
                break

    async def __message_handler(self, message: str) -> None:
        try:
            message = json.loads(message)
            log.info(f"Message received: {message}")
            if message["type"] == "GENERATE_AUDIO_REQUEST":
                text = message["message"]["text"]
                log.info(f"Text to generate audio from: {text}")
                # Check that text isn't empty and doesn't only contain whitespace
                if text.strip() and text:
                    await self.__queue.put(text)
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
