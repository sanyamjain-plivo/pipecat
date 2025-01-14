import os
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
import asyncio
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketTransport,FastAPIWebsocketParams
from plivo import PlivoFrameSerializer
from functions.order_crud import create_order, cancel_order, get_order, add_item_to_order, delete_item_from_order, update_quantity_of_item_in_order
from pipecat.processors.aggregators.llm_response import LLMAssistantResponseAggregator,LLMUserResponseAggregator
from rag.plivo_langchain_processor import PlivoLangchainProcessor
from wrappers.plivo_deepgram_wrapper import PlivoDeepgramWrapper
from wrappers.plivo_openai_wrapper import PlivoOpenAIWrapper
from utils.helpers import handle_transport_event
from dotenv import load_dotenv

load_dotenv(dotenv_path="env", override=True)

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

    llm = PlivoOpenAIWrapper(llm_model="gpt-4o", pinecone_index_name="drive-thru-menu", retriever_tool_name="kFC_order_menu", retriever_tool_description="Useful when you need to answer questions about KFC Menu")
    lc_tools = [llm.create_retriever_tool(), create_order, cancel_order, get_order, add_item_to_order, delete_item_from_order, update_quantity_of_item_in_order]
    llm.set_llm_tools(lc_tools)
    lc = PlivoLangchainProcessor(llm=llm)
    
    plivo_deepgram = PlivoDeepgramWrapper()

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            plivo_deepgram.get_stt_service(), 
            LLMUserResponseAggregator(),
            lc,  # LLM
            plivo_deepgram.get_tts_service(), 
            transport.output(),  # Websocket output to client
            LLMAssistantResponseAggregator()
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    lc.add_task(task)

    transport.event_handler("on_client_connected")(
        lambda t, c: asyncio.create_task(handle_transport_event("on_client_connected", t, c, task))
    )
    transport.event_handler("on_client_disconnected")(
        lambda t, c: asyncio.create_task(handle_transport_event("on_client_disconnected", t, c, task))
    )
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


