"""Run as stdin from each source checkout with an explicit Python interpreter."""
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import kornia
from benchmarks.common import run_metadata, time_us, warm_up_cpu
from kornia.enhance import integral_image, integral_tensor

torch.set_num_threads(1)
torch.manual_seed(0)
print('python', sys.executable, 'kornia', kornia.__file__, flush=True)
assert Path(kornia.__file__).resolve().parent.parent == Path.cwd().resolve()
warm_up_cpu()
rows = []

def record(name, tensor, dim=None, weights=None):
    def fn():
        output = integral_image(tensor) if dim is None else integral_tensor(tensor, dim)
        if weights is not None:
            return torch.autograd.grad(output, tensor, weights)[0]
        return output

    # High-precision public API context; the regression tests separately use
    # analytical literals, without depending on the implementation as an oracle.
    reference_input = tensor.detach().double().requires_grad_(weights is not None)
    reference = integral_image(reference_input) if dim is None else integral_tensor(reference_input, dim)
    if weights is not None:
        reference = torch.autograd.grad(reference, reference_input, weights.double())[0]
    reference = reference.to(tensor.dtype)
    fn()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as prof:
        output = fn()
    allocs = [e.self_cpu_memory_usage for e in prof.events() if e.self_cpu_memory_usage > 0]
    median, iqr = time_us(fn, min_run_time=.15)
    actual = output.detach()
    finite_reference = torch.isfinite(reference)
    finite_comparison = finite_reference & torch.isfinite(actual)
    error = (actual.double() - reference.double()).abs()[finite_comparison]
    rows.append(dict(
        case=name, shape=list(tensor.shape), dtype=str(tensor.dtype), dim=dim,
        mode='backward' if weights is not None else 'forward',
        median_us=median, iqr_us=iqr, allocation_bytes=sum(allocs), allocation_events=len(allocs),
        output_dtype=str(actual.dtype), nan_count=int(torch.isnan(actual).sum()),
        nonfinite_where_reference_finite=int((finite_reference & ~torch.isfinite(actual)).sum()),
        finite_max_abs_error=float(error.max()) if error.numel() else 0.0,
        last_value=float(actual.flatten()[-1]) if actual.numel() and math.isfinite(float(actual.flatten()[-1])) else None,
        first_value=float(actual.flatten()[0]) if actual.numel() and math.isfinite(float(actual.flatten()[0])) else None,
        sha256=hashlib.sha256(actual.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
    ))

for dtype in (torch.float16, torch.bfloat16):
    eps = torch.finfo(dtype).eps
    length = int(2 / eps) + 1
    cancellation = torch.tensor([1.0, -(1.0-eps)], dtype=dtype).expand(length, 2)
    record('cancellation', cancellation)
    record('reverse_axes_control', cancellation, (1, 0))
    tensor = torch.zeros(2, length, dtype=dtype, requires_grad=True)
    weights = torch.tensor([[1.0], [-(1.0-eps)]], dtype=dtype).expand(2, length)
    record('gradient_cancellation', tensor, weights=weights)
    record('small_image', torch.arange(9, dtype=dtype).reshape(3, 3) / 8)

record('volume_cancellation', torch.tensor([1.0, -1.0], dtype=torch.float16).expand(256, 256, 2), (0, 1, 2))
image = torch.rand(1, 3, 128, 128) * 2 - 1
for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
    data = image.to(dtype)
    record('image_128', data)
    record('single_axis_control', data, (-1,))
    record('empty_batch', data[:0])

result = dict(
    metadata=run_metadata(torch.device('cpu')),
    python=sys.executable, kornia_path=kornia.__file__,
    source_sha256=hashlib.sha256(Path('kornia/enhance/integral.py').read_bytes()).hexdigest(),
    allocation_metric='sum of positive self_cpu_memory_usage from torch.profiler; cumulative bytes, not peak',
    reference='same public API on float64 inputs, cast to input dtype; test assertions use independent analytical literals',
    rows=rows,
)
Path(sys.argv[1]).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(rows, indent=2), flush=True)
