import asyncio
import json
from typing import Optional

import websockets.asyncio
from websockets import ConnectionClosedOK, ConnectionClosedError

from speech_recognition.utils.logger_helper import LoggerHelper

log = LoggerHelper(__name__).get_logger()


class WebSocketClient:
    """A WebSocket client for connecting to a WebSocket server, sending and receiving messages,
    and handling text-to-audio generation requests.

    This client maintains an asynchronous connection with automatic reconnection logic.
    Received messages can be parsed and added to an internal queue for further processing.

    Attributes:
        __uri (str): URI of the WebSocket server to connect to.
        __queue (asyncio.Queue): Queue used for communication between this client and other components.
        __register_message (Optional[str]): Optional message to send immediately after connecting.
        __receive_task (asyncio.Task): Async task responsible for handling incoming messages.
        __ws (ClientConnection): The current WebSocket connection object.
    """

    __register_message = None

    __receive_task = None

    def __init__(self, uri: str, queue: asyncio.Queue) -> None:
        self.__uri = uri
        self.__queue = queue
        self.__ws = None

    async def connect(self, message: Optional[str] = None) -> None:
        """Establishes a connection to the WebSocket server.

        If a registration message is provided, it will be sent immediately after the connection is established.

        Args:
            message (Optional[str]): An optional registration or initialization message.
        """
        log.info(f"Connecting to websocket server at {self.__uri}")
        self.__register_message = message
        try:
            await self.__connect_internal()
        except Exception as e:
            log.error(f"Failed Initial connect to websocket server: {e}")
            await self.__reconnect()

    async def __connect_internal(self) -> None:
        """Internal method to establish connection and start listening for incoming messages."""
        self.__ws = await websockets.connect(self.__uri)

        if self.__register_message is not None:
            log.info(f"Registering with message: {self.__register_message}")
            await self.send_message(self.__register_message)

        if self.__receive_task is not None:
            try:
                self.__receive_task.cancel()
            except asyncio.CancelledError:
                await self.__receive_task

        self.__receive_task = asyncio.create_task(self.__receive_messages())

    async def close_connection(self, message: Optional[str] = None) -> None:
        """Closes the WebSocket connection, optionally sending a final message first.

        Args:
            message (Optional[str]): Optional message to send before closing the connection.
        """
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

    async def send_message(self, message: str | dict) -> None:
        """Sends a message to the WebSocket server.

        Args:
            message (str | dict): The message to send. If a dict is provided, it will be converted to JSON.
        """
        if self.__ws:
            try:
                log.info(f"Sending message: {message}")
                match message:
                    case dict():
                        await self.__ws.send(json.dumps(message))
                    case str():
                        await self.__ws.send(message)
            except Exception as e:
                log.error(f"Error sending message: {e}")
        else:
            log.error("Not connected to the WebSocket server.")

    async def __receive_messages(self) -> None:
        """Listens for incoming messages from the WebSocket server.

        Messages will be logged or handled via the internal message handler.
        """
        while True:
            try:
                raw = await self.__ws.recv()
                if raw == "Hallo Spracherkennung":
                    log.info("Successfully registered with server")
                elif self.__message_handler:
                    await self.__message_handler(raw)
                else:
                    log.info(f"Message received: {raw}")
            except ConnectionClosedOK:
                break
            except ConnectionClosedError:
                log.error("Connection closed by the WebSocket server")
                await self.__reconnect()
                break
            except Exception as e:
                log.error(f"Exception occurred while awaiting messages: {e}")
                break

    async def __message_handler(self, message: str) -> None:
        """Handles a specific format of WebSocket message, particularly GENERATE_AUDIO_REQUEST.

        Args:
            message (str): A JSON-formatted string message from the server.
        """
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

    async def __reconnect(self) -> None:
        """Handles reconnection attempts in case the connection is lost."""
        while True:
            try:
                log.info("Attempting to reconnect")
                await asyncio.sleep(5)
                await self.__connect_internal()
                log.info("Reconnected")
                break
            except Exception as e:
                log.warning(f"Reconnect attempt failed: {e}")
