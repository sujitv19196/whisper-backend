import json 

import whisper_timestamped as whisper
from .util import get_device

def transcribe(filepath): 
    device = get_device()
    model = "tiny"
    # TODO change model based on device
    # TODO load model from file
    audio = whisper.load_audio(filepath)

    model = whisper.load_model(model, device="cpu")

    result = whisper.transcribe(model, audio, vad=True)
    result = json.dumps(result, indent = 2, ensure_ascii = False)
    print(result)
    return result