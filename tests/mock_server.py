import asyncio
import json
import threading
import traceback

from websockets import ConnectionClosed
from websockets.asyncio.server import serve

from speech_recognition import LoggerHelper

log = LoggerHelper(__name__).get_logger()


async def consumer_handler(websocket, queue):
    # Handle incoming messages from the client
    try:
        async for message in websocket:
            log.debug(f"Received from client: {message}")
            if queue is not None:
                await queue.put(message)
                log.debug(f"Queued message: {message}")

    except ConnectionClosed:
        log.info("Consumer: Connection closed.")


async def produce(queue):
    # This function runs input() in a thread so it doesn’t block the event loop.
    if queue is None:
        text = await asyncio.to_thread(input, "Send a message to the client: ")
    else:
        text = await queue.get()
        if text is None:  # Sentinel value for shutdown
            return None
    return json.dumps({"type": "GENERATE_AUDIO_REQUEST", "message": {"text": text}})


async def producer_handler(websocket, queue):
    # Handle outgoing messages to the client
    while True:
        try:
            message = await produce(queue)
            if message is None:  # Handle shutdown
                break
            await websocket.send(message)
        except ConnectionClosed:
            print("Producer: Connection closed.")
            break


async def handler(websocket, in_queue, out_queue):
    try:
        # Use asyncio.gather to run producer and consumer concurrently.
        consumer_task = asyncio.create_task(consumer_handler(websocket, out_queue))
        producer_task = asyncio.create_task(producer_handler(websocket, in_queue))

        # Wait for either to finish (which happens on connection closure)
        done, pending = await asyncio.wait(
            [consumer_task, producer_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel any pending tasks
        for task in pending:
            task.cancel()
    except Exception as e:
        log.error(f"{e}")
        traceback.print_exc()
        raise e


def create_handler(in_queue, out_queue):
    async def wrapped_handler(websocket):
        await handler(websocket, in_queue, out_queue)

    return wrapped_handler


async def start_mock_server(host="localhost", port=8080, in_queue=None, out_queue=None):
    log.info("Server starting at ws://localhost:8080")
    try:
        server = await serve(create_handler(in_queue, out_queue), host, port)
        return server
    except Exception as e:
        log.error(f"{e}")
        traceback.print_exc()
        raise e


def serve_mock(
    host="localhost", port=8080, in_queue=None, out_queue=None, shutdown_event=None
):
    async def run_server():
        server = await start_mock_server(host, port, in_queue, out_queue)
        while True:
            await asyncio.sleep(1)  # Keep the server alive
            if shutdown_event.is_set():
                server.close()

    loop = asyncio.new_event_loop()
    loop.run_until_complete(run_server())


def start_server_thread(
    in_queue=asyncio.Queue(), out_queue=asyncio.Queue(), shutdown_event=None
):
    thread = threading.Thread(
        target=serve_mock,
        kwargs={
            "in_queue": in_queue,
            "out_queue": out_queue,
            "shutdown_event": shutdown_event,
        },
        daemon=True,
    )
    thread.start()
    return thread


if __name__ == "__main__":
    serve_mock()
