"""Smoke job. Mode gpu burns the GPU with logging, mode silent sleeps without output."""

import json
import os
import sys
import time

mode = sys.argv[1] if len(sys.argv) > 1 else "gpu"
os.makedirs("/content/run/out", exist_ok=True)
if mode == "silent":
    time.sleep(3600)
else:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
    x = torch.randn(8192, 8192, device="cuda")
    t0 = time.time()
    step = 0
    while time.time() - t0 < 90:
        for _ in range(20):
            x = torch.tanh(x @ x)
        torch.cuda.synchronize()
        step += 1
        print(f"step {step} elapsed {time.time() - t0:.1f}s", flush=True)
    json.dump({"steps": step, "seconds": time.time() - t0}, open("/content/run/out/result.json", "w"))
    print("finished", flush=True)
