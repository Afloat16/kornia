# Spatial gradient 3D buffer validation

Candidate commit: `a11af2d06bd4f5d5be54d486f966caf4db9ee1b1`.
Base commit: `7ecfbb392285e03f73b4dd4cce62fc3fb8349c33`.

This record accompanies a draft contribution. The full compile and ONNX
importer selections are pending CI. An initial local run was interrupted
because the shared host became unresponsive. A bounded diagnostic sampled
native Torch and ONNX library loading before tests began; the four focused
spatial_gradient3d ONNX conformance cases then passed after initialization.
A collected test is not counted as a passed test.

Completed: 100 complete Sobel module CPU float32/float64 tests, 14 focused
spatial_gradient3d float16/bfloat16 tests, four focused ONNX conformance cases,
full pre-commit, ty 0.0.9, and changelog preview. Python 3.12.14,
PyTorch 2.15.0+cpu, onnxruntime 1.32.0. Direct isolated-environment commands
were used because Pixi was unavailable locally.

`probe.py` runs from each checkout as stdin, using an explicit interpreter,
so the local source import can be verified. Its checks cover 48 combinations
of four floating dtypes, singleton/empty/ordinary shapes, and contiguous,
transposed or expanded layouts. Output, input gradient and input tensors all
compared equal with torch.equal; their version counters were unchanged.
`exact-comparison.json` independently checks identical serialized tensor
metadata and storage bytes. The candidate also passed gradcheck,
functionalization, and forward-mode JVP. Script/vmap failures reproduced
unchanged on both revisions and are recorded in the checks JSON.

The profiler traces retain the output at the end of each measured call.
Memory events were independently summed in timestamp order and the final
net live bytes equal the output size in all six traces. Cumulative positive
allocations drop by exactly one complete output tensor, with one fewer
allocation. The measured peak reduction is smaller because the padded input
and difference temporaries remain live during portions of the call.
These are tensor allocation measurements, not process RSS or allocator
reserved-memory measurements. Shared-host timings were unstable; no latency
claim is made.

```sh
/path/to/python - checks /path/to/results < /path/to/probe.py
/path/to/python - memory /path/to/results < /path/to/probe.py
```

`run-conformance.py` runs the unchanged repository test selections while
bounding Torch and ONNX Runtime CPU thread pools. It sets the documented
`ORT_DISABLE_TELEMETRY=1` before import; the runtime API alone does not prevent
its initial telemetry event. For the focused run:

```sh
ORT_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 python - onnx -k spatial_gradient3d --maxfail=1 --timeout=90 -vv < /path/to/run-conformance.py
```

Dropping the `-k` option runs the full selected ONNX packages. Use `compile`
as the first argument for the full compile selection. These full runs are
still required before marking the draft ready; no skip/xfail manifests were
modified to pass a gate.
