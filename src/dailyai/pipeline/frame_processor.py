from abc import abstractmethod
from typing import AsyncGenerator

from dailyai.pipeline.frames import ControlQueueFrame, QueueFrame

"""
This is the base class for all frame processors. Frame processors consume a frame
and yield 0 or more frames. Generally frame processors are used as part of a pipeline,
where frames come from a source queue, are processed by a series of frame processors,
then placed on a sink queue.

By convention, FrameProcessors should immediately yield any frames they don't process.

Stateful FrameProcessors should watch for the EndStreamQueueFrame and finalize their
output, eg. yielding an unfinished sentence if they're aggregating LLM output to full
sentences. EndStreamQueueFrame is also a chance to clean up any services that need to
be closed, del'd, etc.
"""

class FrameProcessor:
    @abstractmethod
    async def process_frame(
        self, frame: QueueFrame
    ) -> AsyncGenerator[QueueFrame, None]:
        if isinstance(frame, ControlQueueFrame):
            yield frame

    @abstractmethod
    async def finalize(self) -> AsyncGenerator[QueueFrame, None]:
        # This is a trick for the interpreter (and linter) to know that this is a generator.
        if False:
            yield QueueFrame()

    @abstractmethod
    async def interrupted(self) -> None:
        pass
