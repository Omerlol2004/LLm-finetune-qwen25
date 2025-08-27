import torch, sys, os
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    i = torch.cuda.current_device()
    print("gpu name:", torch.cuda.get_device_name(i))
    free, total = torch.cuda.mem_get_info()
    print("vram free/total (MB):", free//(1024**2), "/", total//(1024**2))
try:
    import bitsandbytes as bnb
    print("bitsandbytes:", bnb.__version__)
except Exception as e:
    print("bitsandbytes import FAILED:", e)