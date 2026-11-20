# Translated Boxes area: correctness and measured cost

The public `kornia.geometry.boxes.Boxes.compute_area` method evaluates the shoelace formula after angular sorting of the four stored vertices. Multiplying absolute coordinates loses small areas at ordinary image offsets, before any inclusive/exclusive interpretation is involved.

## Reproductions

Directly stored vertices `[(300,300),(310,300),(310,310),(300,310)]` have polygon area 100. The unchanged method returns NaN in float16 and 0 in bfloat16. All coordinates and the expected area are representable in both formats. For float32, `[(8192,8192),(8193,8192),(8193,8193),(8192,8193)]` has area 1 but returns 0. These are image-coordinate examples, not a request for support at arbitrary extreme magnitudes.

The fix promotes half-precision working coordinates to float32, translates coordinates by the first vertex, keeps the existing angular sort and shoelace formula, and casts the result to the original dtype. The four original failures cover two low-precision area/gradient cases, the float32 one-pixel area, and bfloat16 filtering of a valid translated box. The same selection has six passing controls before the fix and all ten passing after it. The full existing boxes test file passed 1,585 cases with eight existing expected failures and two deselections on float16, bfloat16, float32 and float64. Independent literal square/trapezoid/parallelogram areas and gradients, float64 gradcheck, empty temporal batches and non-contiguous batches are included.

This change keeps the existing vertex convention, including the documented inclusive-vertices area mismatch and its expected-failure tests. It is separate from the coordinate-convention migration in upstream #6824/#6710. It does not make already-rounded input edge lengths recoverable, and a true area outside the result dtype's range can still overflow.

## Cost probe

`box_area_bench.py` runs from each checkout root with an explicit interpreter, prints both the interpreter and `kornia.__file__`, uses `benchmarks/common.py` warm-up/timing/metadata helpers, and records the precise source SHA-256. The inputs are one or 1,024 copies of the first square above, in four floating dtypes, for inference or forward plus reverse differentiation. CPU threads are limited to one. Each timing uses `blocked_autorange` and includes median and IQR.

The base is upstream `7ecfbb392285e03f73b4dd4cce62fc3fb8349c33`. The branch JSON identifies the dirty worktree and source hash measured before two explanatory comment lines were added; executable code is the same as the proposed fix. Float32/float64 output and gradient hashes match base for every case. Half-precision output/gradient hashes intentionally change because the old method has the reported defect.

For 1,024 float32 boxes, inference positive tensor allocation traffic rises from 234,552 to 267,320 bytes (+32,768, one coordinate buffer); forward plus reverse rises from 459,844 to 566,340 bytes. For float16/bfloat16, the corresponding totals rise from 207,926 to 302,136 and from 353,340 to 621,632 bytes because working arithmetic is promoted. These sums of positive profiler events are not peak live memory or RSS.

The correction costs additional arithmetic and storage. On this shared CPU, 1,024-box float32 inference measured 142.6 to 190.4 microseconds and forward plus reverse 395.0 to 490.7 microseconds; float16 inference measured 203.7 to 299.3 microseconds. Some controls went in the other direction and some IQRs were large. All samples are retained. This is a correctness repair, not a latency improvement; no end-to-end application benchmark or accelerator result is claimed.

The harness is self-contained apart from Kornia, PyTorch and the repository benchmark helpers. Run it from the corresponding worktree root, for example:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /path/to/python - result.json < /path/to/box_area_bench.py
```

The profiler also emits per-case trace files next to the chosen result JSON. Those temporary traces are not included in the library PR or this evidence branch.
