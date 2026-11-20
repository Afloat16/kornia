"""Measure public polygon drawing on identical inputs before and after coordinate staging."""

import hashlib
import json
import sys
from pathlib import Path

import torch

import kornia
from benchmarks import common

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(92)
print(sys.executable, kornia.__file__, flush=True)
common.warm_up_cpu()
rows = []
for name, coordinate_dtype in (
    ("uniform_float32", torch.float32),
    ("uniform_float16", torch.float16),
    ("uniform_int64", torch.int64),
    ("uniform_float64", torch.float64),
    ("mixed_int64_float32", torch.int64),
    ("tensor_control", torch.float32),
):
    image = torch.zeros(8, 1, 32, 32)
    color = torch.ones(1)
    polygons = []
    for i in range(8):
        mixed = name == "mixed_int64_float32" and i % 2 == 1
        left = 2.5 if mixed else 2
        vertices = [[left, 0], [28, 0], [28, 28], [left, 28]]
        if i % 2 and name != "tensor_control":
            vertices.insert(1, [16, 0])
        polygons.append(torch.tensor(vertices, dtype=torch.float32 if mixed else coordinate_dtype))
    if name == "tensor_control":
        polygons = torch.stack(polygons)

    def call():
        return kornia.image.draw_convex_polygon(image, polygons, color)

    median, iqr = common.time_us(call, min_run_time=0.2)
    output = call()
    digest = hashlib.sha256(output.numpy().tobytes()).hexdigest()
    trace = Path(sys.argv[1]).with_suffix(f".{name}.memory.json")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profile:
        call()
    profile.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    allocations = [e["args"]["Bytes"] for e in events if e.get("name") == "[memory]" and e["args"]["Bytes"] > 0]
    rows.append(dict(case=name, median_us=median, iqr_us=iqr, positive_bytes=sum(allocations),
                     positive_allocations=len(allocations), output_sha256=digest,
                     filled_pixels_per_image=output.sum(dim=(1, 2, 3)).tolist()))

metadata = common.run_metadata(torch.device("cpu"))
metadata["sys_executable"] = sys.executable
metadata["kornia_file"] = kornia.__file__
source = Path(kornia.__file__).parent / "image" / "draw.py"
metadata["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
metadata["image_shape"] = [8, 1, 32, 32]
metadata["image_dtype"] = "float32"
metadata["polygons"] = "alternate 4/5 vertices (4 for tensor control), deterministic rectangles"
common.save_json(sys.argv[1], metadata, rows)
print(json.dumps(rows), flush=True)
