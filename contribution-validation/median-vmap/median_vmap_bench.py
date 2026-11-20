"""Public median-filter cost and output probes, run from each worktree root."""

import hashlib
import json
import sys
from pathlib import Path

import torch

import kornia
from benchmarks import common

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print(sys.executable, kornia.__file__, flush=True)
common.warm_up_cpu()
rows = []
for mapped in (False, True):
    for gradients in (False, True):
        # Deterministic finite values; padding repeats the same source pixel.
        torch.manual_seed(71)
        sample = torch.rand(2, 1, 128, 128).requires_grad_(gradients)
        op = lambda image: kornia.filters.median_blur(image, 3)
        if mapped:
            op = torch.vmap(lambda image: kornia.filters.median_blur(image.unsqueeze(0), 3).squeeze(0))

        def call():
            output = op(sample)
            if gradients:
                return output, torch.autograd.grad(output.sum(), sample)[0]
            return (output,)

        median, iqr = common.time_us(call, min_run_time=0.3)
        output = call()
        hashes = [hashlib.sha256(item.detach().numpy().tobytes()).hexdigest() for item in output]
        trace = Path(sys.argv[1]).with_suffix(f'.{int(mapped)}-{int(gradients)}.memory.json')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profile:
            output = call()
        profile.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())['traceEvents']
        allocations = [e['args']['Bytes'] for e in events if e.get('name') == '[memory]' and e['args']['Bytes'] > 0]
        rows.append(dict(mapped=mapped, gradients=gradients, median_us=median, iqr_us=iqr,
                         positive_bytes=sum(allocations), positive_allocations=len(allocations), hashes=hashes))
        del output

metadata = common.run_metadata(torch.device('cpu'))
metadata['sys_executable'] = sys.executable
metadata['kornia_file'] = kornia.__file__
source = Path(kornia.__file__).parent / 'filters' / 'median.py'
metadata['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
common.save_json(sys.argv[1], metadata, rows)
print(json.dumps(rows), flush=True)
