import psutil
import time
import csv
from dataclasses import dataclass, fields, asdict
import argparse

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


parser = argparse.ArgumentParser(description="Sample CPU and RAM usage to a CSV file.")
parser.add_argument("--output", "-o", default="probe.csv", help="Output CSV file (default: probe.csv)")
parser.add_argument("--interval", "-i", type=float, default=0.1, help="Sampling interval in seconds (default: 0.1)")
parser.add_argument("--duration", "-d", type=float, default=10.0, help="Total duration of sampling in seconds (default: 10.0)")
args = parser.parse_args()

with open(args.output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(Sample)])
    writer.writeheader()

    interval = args.interval
    duration = args.duration
    n_samples = int(duration / interval)
    start = time.perf_counter()
    next_tick = start    

    try:
        for _ in range(n_samples):
            cpu, ram = sample_once()
            t = time.perf_counter() - start
            sample = Sample(t, cpu, ram)
            writer.writerow(asdict(sample))
            next_tick += interval
            time.sleep(max(0, next_tick - time.perf_counter()))
    except KeyboardInterrupt:
        print("stopped")
    finally:
        print("done")