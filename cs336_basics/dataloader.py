import torch
import numpy as np

def load_data(arr, batch_size, context_length, device):

    starts = np.random.randint(0, len(arr) - context_length, size=batch_size)
    idx = starts[:, None] + np.arange(context_length+1)
    window = torch.from_numpy(arr[idx].astype(np.int64))
    if torch.device(device).type == "cuda":
        window = window.pin_memory().to(device, non_blocking=True)
    else:
        window = window.to(device)
    return window[:, :-1], window[:, 1:]
