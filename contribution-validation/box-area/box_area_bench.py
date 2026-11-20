"""Compare the public Boxes area method on identical representable image coordinates."""

import hashlib
import json
import math
import sys
from pathlib import Path

import torch

import kornia
from benchmarks import common

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(81)
print(sys.executable, kornia.__file__, flush=True)
common.warm_up_cpu()
rows = []
for count in (1, 1024):
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        for gradients in (False, True):
            sample = torch.tensor([[[300, 300], [310, 300], [310, 310], [300, 310]]], dtype=dtype)
            sample = sample.repeat(count, 1, 1).requires_grad_(gradients)

            def call():
                result = kornia.geometry.boxes.Boxes(sample).compute_area()
                if gradients:
                    return result, torch.autograd.grad(result.sum(), sample)[0]
                return (result,)

            median, iqr = common.time_us(call, min_run_time=0.2)
            output = call()
            hashes = [hashlib.sha256(x.detach().to(torch.float64).numpy().tobytes()).hexdigest() for x in output]
            values = [[v if math.isfinite(v) else str(v) for v in x.detach().reshape(-1)[:8].to(torch.float64).tolist()]
                      for x in output]
            trace = Path(sys.argv[1]).with_suffix(f'.{count}-{dtype}-{int(gradients)}.memory.json')
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profile:
                output = call()
            profile.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())['traceEvents']
            allocations = [e['args']['Bytes'] for e in events if e.get('name') == '[memory]' and e['args']['Bytes'] > 0]
            rows.append(dict(count=count, dtype=str(dtype), gradients=gradients, median_us=median, iqr_us=iqr,
                             positive_bytes=sum(allocations), positive_allocations=len(allocations),
                             hashes=hashes, first_values=values))
            del output

metadata = common.run_metadata(torch.device('cpu'))
metadata['sys_executable'] = sys.executable
metadata['kornia_file'] = kornia.__file__
source = Path(kornia.__file__).parent / 'geometry' / 'boxes.py'
metadata['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
common.save_json(sys.argv[1], metadata, rows)
print(json.dumps(rows), flush=True)
