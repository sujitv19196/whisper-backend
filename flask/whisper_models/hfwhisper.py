import os
from util import get_device
import torch
from transformers import AutoProcessor, pipeline

# def transcribe(filepath, device="cpu"):
#     pipe = pipeline(
#         "automatic-speech-recognition",
#         model="openai/whisper-small", # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
#         torch_dtype=torch.float16,
#         device=device, # or mps for Mac devices
#         model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
#     )

#     outputs = pipe(
#         filepath,
#         chunk_length_s=30,
#         batch_size=24,
#         return_timestamps=True,
#     )

#     return outputs

# Insanely Fast Whisper parameters 
def transcribe(filepath):
    model = "openai/whisper-large-v3"
    processor = AutoProcessor.from_pretrained(model)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        torch_dtype=torch.float16,
        chunk_length_s=30,
        max_new_tokens=128,
        batch_size=24,
        device=get_device(),
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        model_kwargs={"use_flash_attention_2": True},
        generate_kwargs={"language": "english", "return_timestamps": "True"},
    )
    result = pipe(filepath, return_timestamps=True)

    print(result)
    return result
