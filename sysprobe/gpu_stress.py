import sys
import time
import torch

duration = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0

a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

end = time.perf_counter() + duration
while time.perf_counter() < end:
    for _ in range(50):
        a @ b
    torch.cuda.synchronize()