"""Measure the public AEPE API unchanged on each checkout."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

import kornia
from benchmarks import common

parser = argparse.ArgumentParser()
parser.add_argument("--json", required=True)
args = parser.parse_args()
torch.manual_seed(0)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
common.warm_up_cpu()
print(kornia.__file__, sys.executable, flush=True)
rows = []
for size in (8, 512, 1024):
    for gradients in (False, True):
        flow = torch.randn(1, size, size, 2)
        target = torch.randn_like(flow)
        target[:, 0] = flow[:, 0]
        flow.requires_grad_(gradients)
        target.requires_grad_(gradients)

        def call():
            value = kornia.metrics.aepe(flow, target)
            if gradients:
                return torch.autograd.grad(value, (flow, target))
            return value

        us, iqr = common.time_us(call, min_run_time=0.3)
        rows.append({"shape": list(flow.shape), "gradients": gradients, "median_us": us, "iqr_us": iqr})

for size in (512, 1024):
    flow = torch.randn(1, size, size, 2)
    target = torch.randn_like(flow)
    kornia.metrics.aepe(flow, target, reduction="none")
    trace = Path(args.json).with_suffix(f".memory-{size}.json")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profile:
        value = kornia.metrics.aepe(flow, target, reduction="none")
    profile.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    memory = sorted((e for e in events if e.get("name") == "[memory]"), key=lambda e: e["ts"])
    net = peak = positive = count = 0
    for event in memory:
        size_bytes = event["args"]["Bytes"]
        net += size_bytes
        peak = max(peak, net)
        if size_bytes > 0:
            positive += size_bytes
            count += 1
    rows.append(
        {
            "shape": list(flow.shape),
            "memory_profile": True,
            "positive_bytes": positive,
            "peak_net_bytes": peak,
            "positive_allocations": count,
            "retained_bytes": net,
            "output_bytes": value.numel() * value.element_size(),
        }
    )
    del value

metadata = common.run_metadata(torch.device("cpu"))
metadata["kornia_module"] = str(Path(kornia.__file__).resolve())
source = Path(kornia.__file__).parent / "metrics" / "endpoint_error.py"
metadata["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
common.save_json(args.json, metadata, rows)
print(json.dumps(rows, indent=2))
