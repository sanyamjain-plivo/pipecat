from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.task import PipelineTask
from loguru import logger
import sys

messages = []
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")
async def handle_transport_event(event_type: str, transport, client, task: PipelineTask):
    match event_type:
        case "on_client_connected":
            print("client connected")
            # Kick off the conversation.
            messages.append({"role": "system", "content": "Fetch the KFC menu and simultaneously introduce yourself to the user but do not share the menu."})
            await task.queue_frames([LLMMessagesFrame(messages)])
        case "on_client_disconnected":
            await task.queue_frames([EndFrame()])
        case _:
            logger.warning(f"Unhandled transport event: {event_type}")