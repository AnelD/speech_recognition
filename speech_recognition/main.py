import asyncio
import json
import threading
from pathlib import Path

from speech_recognition import (
    LoggerHelper,
    WebSocketClient,
    LLMService,
    ASRService,
    text_to_speech,
    FileSystemObserver,
    config,
)

log = LoggerHelper(__name__).get_logger()

speech_event = asyncio.Event()
llm_event = asyncio.Event()


async def process_transcript(
    queue: asyncio.Queue, client: WebSocketClient, llm: LLMService
) -> None:
    """Process text prompts into structured JSON using LLM."""
    while True:
        request = await queue.get()
        await handle_llm_request(request, client, llm)
        llm_event.set()


async def handle_llm_request(
    request: dict, client: WebSocketClient, llm: LLMService
) -> None:
    """Handle a single LLM generation request."""
    prompt = request["prompt"]
    req_type = request["req_type"]

    try:
        log.info(f"Prompt received: {prompt} for {req_type}")
        response = llm.generate_json_response(prompt, req_type)
        data = json.loads(response)
        await client.send_message(
            json.dumps(
                {
                    "type": "EXTRACT_DATA_FROM_AUDIO_SUCCESS",
                    "message": {"text": data},
                }
            )
        )
    except Exception as e:
        log.exception(f"LLM error: {e}")
        await client.send_message(
            json.dumps(
                {
                    "type": "EXTRACT_DATA_FROM_AUDIO_ERROR",
                    "message": {
                        "text": f"Error while processing prompt {prompt} for request {req_type}"
                    },
                }
            )
        )


async def process_audio(
    in_queue: asyncio.Queue,
    out_queue: asyncio.Queue,
    client: WebSocketClient,
    asr: ASRService,
) -> None:
    """Convert incoming audio files into text."""
    # Get configured filepaths, encode, and decode to handle windows paths with \
    in_path = Path(config.AUDIO_IN_DIR.encode("unicode_escape").decode()).resolve()
    out_path = Path(config.AUDIO_OUT_DIR.encode("unicode_escape").decode()).resolve()

    while True:
        request = await in_queue.get()
        await handle_audio_request(request, client, asr, in_path, out_path, out_queue)
        speech_event.set()


async def handle_audio_request(
    request: dict,
    client: WebSocketClient,
    asr: ASRService,
    in_path: Path,
    out_path: Path,
    out_queue: asyncio.Queue,
) -> None:
    """Handle a single audio file transcription."""
    filename = request["filename"]
    req_type = request["req_type"]

    if req_type == "BAD_REQUEST":
        log.exception(f"Bad request: {filename}")
        await client.send_message(
            json.dumps(
                {
                    "type": "EXTRACT_DATA_FROM_AUDIO_ERROR",
                    "message": {"text": f"Bad request for file {filename}"},
                }
            )
        )
        return

    try:
        text = asr.transcribe(
            f"{str(in_path)}/{filename}",
            f"{str(out_path)}/{filename.rsplit('.', 1)[0]}.wav",
        )
        await out_queue.put({"prompt": text, "req_type": req_type})
    except Exception as e:
        log.exception(f"Transcription error for {filename}: {e}")
        await client.send_message(
            json.dumps(
                {
                    "type": "EXTRACT_DATA_FROM_AUDIO_ERROR",
                    "message": {"text": f"Error transcribing file {filename}"},
                }
            )
        )


async def main():
    loop = asyncio.get_running_loop()

    # Queues
    text_queue = asyncio.Queue()
    speech_queue = asyncio.Queue()
    llm_queue = asyncio.Queue()

    # Services
    client = WebSocketClient(config.WEBSOCKET_URI, text_queue)
    asr = ASRService()
    llm = LLMService()

    await client.connect()
    await client.send_message("sp")

    # Tasks
    asyncio.create_task(text_to_speech(text_queue))
    asyncio.create_task(process_audio(speech_queue, llm_queue, client, asr))
    asyncio.create_task(process_transcript(llm_queue, client, llm))

    # Start file system observer in separate thread
    path = str(Path(config.AUDIO_IN_DIR.encode("unicode_escape").decode()).resolve())
    observer = FileSystemObserver(loop, speech_queue)
    threading.Thread(target=observer.start_observer, args=(path,), daemon=True).start()
    log.info(f"File observer watching folder: {path}")

    try:
        while True:
            log.debug("Waiting for both speech and LLM events to complete...")
            await asyncio.gather(speech_event.wait(), llm_event.wait())

            # Reset both events
            speech_event.clear()
            llm_event.clear()

            log.info("Ready for next job")

    # Graceful Shutdown,
    # when closed with, for example, CTRL+C the currently running tasks raise CancelledError
    # The actual KeyboardInterrupt is raised outside the event loop
    except asyncio.CancelledError:
        log.info("Shutdown requested.")
    finally:
        log.info("Cleaning up...")

        # Stop observer
        observer.stop_observer()

        # Disconnect WebSocket client
        await client.close_connection("sp Closing connection")

        log.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
