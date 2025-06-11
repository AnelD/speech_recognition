import asyncio
import logging

import pytest
from watchdog.events import FileCreatedEvent

from speech_recognition.services.file_observer import FileObserver
from speech_recognition.services.llm_service import RequestType


@pytest.fixture(autouse=True)
def disable_logging():
    # Disables logging during tests
    logging.disable(logging.CRITICAL)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("person-test.m4a", RequestType.PERSON_DATA),
        ("command-test.m4a", RequestType.COMMAND),
        ("unknown-test.m4a", RequestType.BAD_REQUEST),
    ],
)
async def test_on_created_real_queue(tmp_path, test_input, expected):
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()
    file_observer = FileObserver(loop, queue, str(tmp_path))

    # Create fake events for every corresponding filename and RequestType
    event = FileCreatedEvent(src_path=str(tmp_path / test_input))
    file_observer.on_created(event)

    await asyncio.sleep(0.1)
    assert not queue.empty(), "Queue should have one item after file creation"
    item = await queue.get()

    assert test_input in item["file"]
    assert item["req_type"] == expected


@pytest.mark.asyncio
async def test_add_to_queue(tmp_path):
    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    file_observer = FileObserver(loop, queue, str(tmp_path))

    test_item = {"test": "value"}

    await file_observer._FileObserver__add_to_queue(test_item)

    assert not queue.empty()
    item = await queue.get()
    assert item == test_item


def test_start_observer(mocker, tmp_path):
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()
    file_observer = FileObserver(loop, queue, str(tmp_path))

    # Patch Observer
    mock_observer = mocker.patch(
        "speech_recognition.services.file_observer.Observer"
    ).return_value

    mock_observer.start.return_value = None
    mock_observer.join.return_value = None

    # Start the observer
    file_observer.start()

    # Assert that things happened
    mock_observer.schedule.assert_called_once_with(
        event_handler=file_observer, path=str(tmp_path), recursive=False
    )
    mock_observer.start.assert_called_once()
    mock_observer.join.assert_called_once()


def test_stop_observer(mocker, tmp_path):
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()
    file_observer = FileObserver(loop, queue, str(tmp_path))

    # Create a mock observer
    mock_observer = mocker.Mock()
    file_observer._FileObserver__observer = mock_observer

    # Stop the observer
    file_observer.stop()

    # Assert that it was actually stopped
    mock_observer.stop.assert_called_once()
    mock_observer.join.assert_called_once()
