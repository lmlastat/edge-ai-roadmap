import psutil
import time
import csv
from dataclasses import dataclass, fields, asdict

@dataclass
class Sample:
    t: float
    cpu_percent: float
    ram_used_MB: float

def sample_once() -> tuple[float, float]:
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().used / 2**20
    return cpu, ram


psutil.cpu_percent(interval=None)  # reset the internal counter


with open("probe.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(Sample)])
    writer.writeheader()
    start = time.perf_counter()
    for _ in range(100):
        cpu, ram = sample_once()
        t = time.perf_counter() - start
        sample = Sample(t, cpu, ram)
        writer.writerow(asdict(sample))
        time.sleep(0.1)