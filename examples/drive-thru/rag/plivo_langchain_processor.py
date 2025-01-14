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
from wrappers.plivo_openai_wrapper import PlivoOpenAIWrapper
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


class PlivoLangchainProcessor(LangchainProcessor):
    
    def __init__(self, llm: PlivoOpenAIWrapper, transcript_key: str = "input"):
        self.llm = llm
        self._transcript_key = transcript_key
        self.message_store = {}
        super().__init__(self.create_chain(), transcript_key)
        
    def add_task(self, task):
        self._task = task

    @staticmethod
    def __get_token_value(text: Union[str, AIMessageChunk], task) -> str:
        match text:
            case str():
                return text
            case AIMessageChunk():
                return text.content
            case dict() as d if "output" in d:
                return d["output"]
            case dict() as d if "actions" in d:
                # this is a tool call
                tool_agent_action = d["actions"]
                tool_agent_action[0].log
                if "responded" in tool_agent_action[0].log:
                    response = tool_agent_action[0].log.split("responded:")[1].split("\n")[0].strip()
                    return response
                return ""
            case _:
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
            
    def create_chain(self):
        
        self._chain = RunnableWithMessageHistory(
            self.llm.get_llm_agent_executor(), 
            self.get_session_history,
            history_messages_key="chat_history",
            input_messages_key="input",
            output_messages_key="output"
        )
        return self._chain
        
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.message_store:
            self.message_store[session_id] = ChatMessageHistory()
        return self.message_store[session_id]





