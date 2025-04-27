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
from speech_recognition.llm_service import RequestType

log = LoggerHelper(__name__).get_logger()

speech_event = asyncio.Event()
llm_event = asyncio.Event()


async def transcript_to_json(
    queue: asyncio.Queue, client: WebSocketClient, llm: LLMService
) -> None:
    while True:
        request = await queue.get()
        prompt = request["prompt"]
        req_type = request["req_type"]
        try:
            log.info(f"Received prompt: {prompt}, for request_type: {req_type}")
            response = llm.generate_json_response(prompt, req_type)
            person = json.loads(response)
            log.info(person)
            message = {
                "type": "EXTRACT_DATA_FROM_AUDIO_SUCCESS",
                "message": {"text": person},
            }
            await client.send_message(json.dumps(message))
        except Exception as e:
            log.exception(f"Error during LLM processing {e}")
            message = {
                "type": "EXTRACT_DATA_FROM_AUDIO_ERROR",
                "message": {"text": f"Error while processing prompt {prompt}"},
            }
            await client.send_message(json.dumps(message))
        finally:
            llm_event.set()


async def speech_to_transcript(
    in_queue: asyncio.Queue,
    out_queue: asyncio.Queue,
    client: WebSocketClient,
    asr: ASRService,
) -> None:
    # Get configured filepaths, encode, and decode to handle windows paths with \
    in_path = str(Path(config.AUDIO_IN_DIR.encode("unicode_escape").decode()).resolve())
    out_path = str(
        Path(config.AUDIO_OUT_DIR.encode("unicode_escape").decode()).resolve()
    )

    while True:
        request = await in_queue.get()
        log.info(f"Received request: {request}")
        filename = request["filename"]
        req_type = request["req_type"]

        if req_type == RequestType.BAD_REQUEST:
            log.warning(f"Bad request: {filename}")
            message = {
                "type": "EXTRACT_DATA_FROM_AUDIO_ERROR",
                "message": {"text": f"Bad request for file {filename}"},
            }
            await client.send_message(json.dumps(message))
            continue

        in_file = f"{in_path}/{filename}"
        out_file = f"{out_path}/{filename.rsplit('.', 1)[0]}.wav"

        log.info(f"Received file for transcription: {filename}")
        try:
            message = {
                "type": "EXTRACT_DATA_FROM_AUDIO_STARTING",
                "message": {"fileName": filename},
            }
            await client.send_message(json.dumps(message))

            text = asr.transcribe(in_file, out_file)
            await out_queue.put({"prompt": text, "req_type": req_type})
        except Exception as e:
            log.exception(f"Error transcribing {filename}: {e}")
            message = {
                "type": "EXTRACT_DATA_FROM_AUDIO_ERROR",
                "message": {"text": f"Error transcribing file {filename}"},
            }
            await client.send_message(json.dumps(message))
        finally:
            speech_event.set()


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
    asyncio.create_task(speech_to_transcript(speech_queue, llm_queue, client, asr))
    asyncio.create_task(transcript_to_json(llm_queue, client, llm))

    # Start file system observer in separate thread
    path = str(Path(config.AUDIO_IN_DIR.encode("unicode_escape").decode()).resolve())
    observer = FileSystemObserver(loop, speech_queue)
    threading.Thread(target=observer.start_observer, args=(path,), daemon=True).start()
    log.info(f"Started file observer on {path}")

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
