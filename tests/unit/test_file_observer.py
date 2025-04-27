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
    file_observer = FileObserver(loop, queue)

    event = FileCreatedEvent(src_path=str(tmp_path / test_input))

    file_observer.on_created(event)

    await asyncio.sleep(0.1)
    assert not queue.empty(), "Queue should have one item after file creation"
    item = await queue.get()

    assert item["filename"] == test_input
    assert item["req_type"] == expected


@pytest.mark.asyncio
async def test_add_to_queue(tmp_path):
    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    file_observer = FileObserver(loop, queue)

    test_item = {"test": "value"}

    await file_observer._add_to_queue(test_item)

    # Verify
    assert not queue.empty()
    item = await queue.get()
    assert item == test_item


def test_start_observer(mocker, tmp_path):
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()
    file_observer = FileObserver(loop, queue)

    # Patch Observer
    mock_observer = mocker.patch(
        "speech_recognition.services.file_observer.Observer"
    ).return_value

    # Act
    mock_observer.start.return_value = None
    mock_observer.join.return_value = None

    file_observer.start_observer(str(tmp_path))

    # Assert
    mock_observer.schedule.assert_called_once_with(
        file_observer, str(tmp_path), recursive=False
    )
    mock_observer.start.assert_called_once()
    mock_observer.join.assert_called_once()


def test_stop_observer(mocker):
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()
    file_observer = FileObserver(loop, queue)

    # Create a mock observer
    mock_observer = mocker.Mock()
    file_observer.observer = mock_observer

    # Act
    file_observer.stop_observer()

    # Assert
    mock_observer.stop.assert_called_once()
    mock_observer.join.assert_called_once()
