from typing import Union
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import Runnable
from loguru import logger
from pipecat.frames.frames import (
    TextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frameworks.langchain import LangchainProcessor

class LangchainRAGProcessor(LangchainProcessor):
    def __init__(self, chain: Runnable, transcript_key: str = "input"):
        super().__init__(chain, transcript_key)
        self._chain = chain
        self._transcript_key = transcript_key
        
    def add_task(self, task):
        self._task = task

    @staticmethod
    def __get_token_value(text: Union[str, AIMessageChunk], task) -> str:
        print("text is", text)
        match text:
            case str():
                print("qwertyuio")
                return text
            case AIMessageChunk():
                print("werfghj")
                return text.content
            case dict() as d if "output" in d:
                print("asdfghjk")
                return d["output"]
            case dict() as d if "actions" in d:
                # this is a tool call
                print("tool call")
                tool_agent_action = d["actions"]
                tool_agent_action[0].log
                if "responded" in tool_agent_action[0].log:
                    response = tool_agent_action[0].log.split("responded:")[1].split("\n")[0].strip()
                    print("tool_agent_action", response)
                    return response
                return ""
            case _:
                print("zxcvbnm")
                return ""

    async def _ainvoke(self, text: str):
        logger.debug(f"Invoking chain with {text}")
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            async for token in self._chain.astream(
                {self._transcript_key: text},
                config={"configurable": {"session_id": self._participant_id}},
            ):
                # await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(TextFrame(self.__get_token_value(token, self._task)))
                # await self.push_frame(LLMFullResponseEndFrame())
        except GeneratorExit:
            logger.warning(f"{self} generator was closed prematurely")
        except Exception as e:
            logger.exception(f"{self} an unknown error occurred: {e}")
            
        finally:
            await self.push_frame(LLMFullResponseEndFrame())


