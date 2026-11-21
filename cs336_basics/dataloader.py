import torch
from einops import rearrange

def load_data(arr, batch_size, context_length, device):
    n = arr.size
    y = torch.randint(0, n-context_length, (batch_size,))
    y = rearrange(y, "b -> b 1")
    offsets = torch.arange(context_length)
    inputs_index = y + offsets
    targets_index = inputs_index + 1
    return torch.tensor(arr[inputs_index].astype("int64")), torch.tensor(arr[targets_index].astype("int64"))
