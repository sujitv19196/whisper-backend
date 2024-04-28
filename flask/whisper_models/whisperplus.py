from whisperplus import SpeechToTextPipeline

def transcribe(filepath, device="cpu"):
    pipeline = SpeechToTextPipeline(model_id="openai/whisper-large-v3")
    transcript = pipeline(filepath, 
                        model_id="openai/whisper-large-v3", 
                        language="english")
    print(transcript)
    return transcript
