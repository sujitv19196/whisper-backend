import torch

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
