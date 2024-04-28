import json 

import whisper_timestamped as whisper
from .util import get_device

def transcribe(filepath): 
    device = get_device()
    model = "openai/whisper-large-v3"
    # TODO change model based on device
    audio = whisper.load_audio(filepath)

    model = whisper.load_model(model, device=device, backend="transformers")

    result = whisper.transcribe(model, audio)
    result = json.dumps(result, indent = 2, ensure_ascii = False)
    print(result)
    return result