import os
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

def transcribe(filepath, device="cpu"):
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small", # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
        torch_dtype=torch.float16,
        device=device, # or mps for Mac devices
        model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
    )

    outputs = pipe(
        filepath,
        chunk_length_s=30,
        batch_size=24,
        return_timestamps=True,
    )

    return outputs
