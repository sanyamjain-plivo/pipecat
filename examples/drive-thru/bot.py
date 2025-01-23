import os
import sys
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from deepgram import LiveOptions
from pipecat.frames.frames import TextFrame
from pipecat.serializers.plivo import PlivoFrameSerializer
from loguru import logger
from dotenv import load_dotenv
from pinecone import Pinecone

from pinecone import Pinecone
from rag.pinecone_index import set_data_to_pinecone
from langchain_pinecone import PineconeVectorStore

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate

from functions.order_crud import create_order, cancel_order, get_order, add_item_to_order, delete_item_from_order, update_quantity_of_item_in_order
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.chains import LLMChain
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)

from rag.helpers import (
    LangchainRAGProcessor,
)





os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["DEEPGRAM_API_KEY"] = os.getenv("DEEPGRAM_API_KEY")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
messages = []
message_store = {}


load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in message_store:
        message_store[session_id] = ChatMessageHistory()
    return message_store[session_id]


async def run_bot(websocket_client, stream_id):
    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            serializer=PlivoFrameSerializer(stream_id),
        ),
    )
    retriever = set_and_retrieve_pinecone_data()
    
    
    answer_prompt = ChatPromptTemplate.from_messages(
    [SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template="""You are Billy, salesperson at KFC. Your job is to take customer orders. \
            Make yourself sound exciting and welcoming to the user. Do not use any emojis or special characters. \
            Use fillers like 'Hmm,' 'Well,' 'You know,' or 'Let me think' occasionally to make the conversation more lifelike. \
            Only share the entire menu along with all the items by category when asked. Mention Prices and Nutritional info only when asked. \
            you need to call different tools and retrive_tool for RAG to answer KFC menu related queries from the provided content only. \
            After sharing the menu, ask what the customer would like to order. After each order, confirm if they want anything else. \
            If they say no, thank them, generate a random order Id between 1-1000 and always pass the information along with total price, items, item price and quantity to the create_order tool. \
            provide the total price and the same order Id to the customer once the order is successfully created by the create_order tool, and end the conversation. \
            Be concise, friendly, and helpful. Use only words, numbers, and essential punctuation like '?' or '!'.
            Answer questions related to orders only. For all other questions, politely say that you are not sure about the answer and ask the customer to ask a question related to orders. \
            Make tool calls with all the parameters required by the toolwhen needed or requested. before making tool calls, add some fillers like 'Sure, let me get the order details for you' or 'Sure, let me create the order for you' or 'Sure, let me check on that for you' or 'I'll get that right away' or 'I'll check on that for you""")),
     MessagesPlaceholder(variable_name='chat_history', optional=True),
     HumanMessagePromptTemplate(prompt=PromptTemplate(
         input_variables=['input'], template='{input}')),
     MessagesPlaceholder(variable_name='agent_scratchpad')])
    
    
    
    llm = ChatOpenAI(model="gpt-4o")
    retriever_tool = create_retriever_tool(
        retriever, 
        name="kFC_order_menu",
        description="""Useful when you need to answer questions about KFC Menu""")
    
    lc_tools = [retriever_tool, create_order, cancel_order, get_order, add_item_to_order, delete_item_from_order, update_quantity_of_item_in_order]
    
    agent = create_openai_tools_agent(llm=llm, tools=lc_tools, prompt=answer_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=lc_tools, verbose=True)

    history_chain = RunnableWithMessageHistory(
        agent_executor, 
        get_session_history,
        history_messages_key="chat_history",
        input_messages_key="input",
        output_messages_key="output"
    )
    
    lc = LangchainRAGProcessor(chain=history_chain)
    tma_in = LLMUserResponseAggregator()
    tma_out = LLMAssistantResponseAggregator()
    
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            encoding="linear16",
            language="en-IN",
            model="nova-2",
            sample_rate=16000,
            channels=1,
            interim_results=False,
            smart_format=True,
            punctuate=True,
            profanity_filter=True,
            vad_events=False,
        ),
    )

    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        sample_rate=16000,
        voice="aura-asteria-en"
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt, 
            tma_in,
            lc,  # LLM
            tts, 
            transport.output(),  # Websocket output to client
            tma_out
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    lc.add_task(task)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        print("client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Fetch the KFC menu and simultaneously introduce yourself to the user but do not share the menu."})
        await task.queue_frames([LLMMessagesFrame(messages)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.queue_frames([EndFrame()])

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)



async def set_and_retrieve_pinecone_data():
    raw_data = open("new_menu.txt", "r").read()
    set_data_to_pinecone(pc, raw_data)
    
    vector_store = PineconeVectorStore.from_existing_index(
        index_name="drive-thru-menu",
        embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")),
    )
    
    return vector_store.as_retriever()
