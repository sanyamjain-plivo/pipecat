import os
import deepgram
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from deepgram import LiveOptions


class PlivoDeepgramWrapper:
    def __init__(self):
        self.deepgram_stt = None
        self.deepgram_tts = None

    def get_stt_service(self):
        # add the options for the stt service
        options = LiveOptions(
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
        )
        self.deepgram_stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"), 
            live_options=options
        )
        return self.deepgram_stt
        
    def get_tts_service(self):
        self.deepgram_tts = DeepgramTTSService(
            api_key=os.getenv("DEEPGRAM_API_KEY"), 
            voice="aura-asteria-en",
            sample_rate=16000,
        )
        return self.deepgram_tts
