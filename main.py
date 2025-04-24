import asyncio
import json
import logging
import pathlib
import threading

import tts
from ASRService import ASRService
from FileSystemObserver import FileSystemObserver
from LLMService import LLMService
from LoggerHelper import LoggerHelper
from WebSocketClient import WebSocketClient

logger = LoggerHelper("app_log", log_level=logging.DEBUG).get_logger()

speech_event = asyncio.Event()
llm_event = asyncio.Event()


async def waitForInput(queue, client, llm: LLMService):
    while True:
        prompt = await queue.get()

        try:
            logger.info(f"Received prompt: {prompt}")
            response = llm.generate_json_response(prompt)
            person = json.loads(response)
            print(person)
            message = {
                "type": "EXTRACT_DATA_FROM_AUDIO_SUCCESS",
                "message": {"text": person},
            }
            await client.send_message(json.dumps(message))
        except Exception as e:
            logger.exception(f"Error during LLM processing {e}")
        finally:
            llm_event.set()


async def speechToJson(in_queue, out_queue, client, asr: ASRService):
    while True:
        filename = await in_queue.get()
        infile = f"data/in/{filename}"
        outfile = f"data/out/{filename.rsplit('.', 1)[0]}.wav"

        logger.info(f"Received file for transcription: {filename}")
        try:
            message = {
                "type": "EXTRACT_DATA_FROM_AUDIO_SUCCESS",
                "message": {"fileName": filename},
            }
            await client.send_message(json.dumps(message))

            text = asr.transcribe(infile, outfile)
            await out_queue.put(text)
        except Exception as e:
            logger.exception(f"Error transcribing {filename}: {e}")
        finally:
            speech_event.set()


async def main():
    loop = asyncio.get_running_loop()

    # Queues
    text_queue = asyncio.Queue()
    speech_queue = asyncio.Queue()
    llm_queue = asyncio.Queue()

    # Services
    client = WebSocketClient("ws://localhost:8080")
    asr = ASRService()
    llm = LLMService()

    await client.connect()
    await client.send_message("sp")

    # Tasks
    asyncio.create_task(tts.text_to_speech(text_queue))
    asyncio.create_task(speechToJson(speech_queue, llm_queue, client, asr))
    asyncio.create_task(waitForInput(llm_queue, client, llm))

    # Start file system observer in separate thread
    path = pathlib.Path("data/in").resolve()
    observer = FileSystemObserver(loop, speech_queue)
    threading.Thread(target=observer.start_observer, args=(path,), daemon=True).start()
    logger.info(f"Started file observer on {path}")

    # Event loop
    while True:
        logger.debug("Waiting for both speech and LLM events to complete...")
        await asyncio.gather(speech_event.wait(), llm_event.wait())

        # Reset both events
        speech_event.clear()
        llm_event.clear()

        logger.info("Ready for next job")


if __name__ == "__main__":
    asyncio.run(main())
