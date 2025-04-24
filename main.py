import asyncio
import logging
import pathlib
import threading

import WebSocketClient as ws
import tts
from ASRService import ASRService
from FileSystemObserver import FileSystemObserver
from LLMService import LLMService
from LoggerHelper import LoggerHelper

logger = LoggerHelper("app_log", log_level=logging.DEBUG).get_logger()


async def waitForInput(queue, client, llm: LLMService):
    while True:
        prompt = await queue.get()

        try:
            logger.info(f"Received prompt: {prompt}")
            response = llm.generate_json_response(prompt)

            await client.send_message(
                f"""
            {{
                "type": "EXTRACT_DATA_FROM_AUDIO_SUCCESS",
                "message": {{
                    "text": {response}
                }}
            }}
            """
            )
        except Exception as e:
            logger.exception(f"Error during LLM processing {e}")
        finally:
            queue.task_done()


async def speechToJson(in_queue, out_queue, client, asr: ASRService):
    while True:
        filename = await in_queue.get()
        infile = f"data/in/{filename}"
        outfile = f"data/out/{filename.rsplit('.', 1)[0]}.wav"

        logger.info(f"Received file for transcription: {filename}")
        try:
            await client.send_message(
                f"""
            {{
                "type": "EXTRACT_DATA_FROM_AUDIO_STARTING",
                "message": {{
                    "text": {{
                        "fileName": "{filename}"
                    }}
                }}
            }}
            """
            )
            text = asr.transcribe(infile, outfile)
            await out_queue.put(text)
        except Exception as e:
            logger.exception(f"Error transcribing {filename}: {e}")
        finally:
            in_queue.task_done()


speech_event = asyncio.Event()
llm_event = asyncio.Event()


async def main():
    loop = asyncio.get_running_loop()

    # Queues
    text_queue = asyncio.Queue()
    speech_queue = asyncio.Queue()
    llm_queue = asyncio.Queue()

    # Services
    client = ws.WebSocketClient("ws://localhost:8080")
    asr = ASRService()
    llm = LLMService()

    # Tasks
    asyncio.create_task(tts.text_to_speech(text_queue))
    asyncio.create_task(speechToJson(speech_queue, llm_queue, client, asr))
    asyncio.create_task(waitForInput(llm_queue, client, llm))

    await client.connect()
    await client.send_message("sp")

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
