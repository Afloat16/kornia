# Median-filter function-transform regression and cost comparison

This extends the distance-transform correction in Kornia PR #6852 with the same
derivative-selection correction in `median_blur`; it is not a separate PR or
independent contribution count. Both call sites use the existing
`kornia.core.utils._needs_derivatives` from upstream main
`7ecfbb392285e03f73b4dd4cce62fc3fb8349c33`.

The unmodified median implementation fails 12 new float32/float64 regression
parameterizations: eight forward-mode calls raise the unsupported
`aten::_unpack_dual` batching-rule error, and four reverse-mode cases choose the
inference sorting network and change the tied-window gradient. The functional
fix retains the backend's ordinary `median` tie derivative, including when a
batching wrapper hides `requires_grad`. The tests cover 3 x 3 and 5 x 5 kernels
and forward derivatives inside `no_grad`.

The public-API benchmark uses seeded float32 `(2, 1, 128, 128)` data, a 3 x 3
median filter with the default reflect padding, and both ordinary and vmapped
callers. The seed is 71. Each sample uses the repository's CPU warmup and timing
helpers. A fresh CPU profiler trace records each mode after warmup and timing.
The metric sums positive `[memory]` allocation events, rather than peak live
memory, process RSS, or retained storage.

| Call | Base allocated bytes | Branch allocated bytes | Base median / IQR (us) | Branch median / IQR (us) |
| --- | ---: | ---: | ---: | ---: |
| Ordinary inference | 6,334,560 | 6,334,560 | 475.07 / 67.61 | 533.90 / 84.03 |
| Ordinary reverse differentiation | 12,280,464 | 12,280,464 | 8,500.50 / 1,178.08 | 7,902.85 / 476.79 |
| Vmapped inference | 6,334,560 | 6,334,560 | 655.87 / 46.34 | 818.27 / 341.69 |
| Vmapped reverse differentiation | 35,226,956 | 12,280,464 | 16,105.24 / 776.42 | 8,133.29 / 399.46 |

All four output digests and both gradient digests match exactly on this finite
random input. It intentionally avoids the tied-gradient discrepancy exercised
by the regression tests. Ordinary-call allocation totals are unchanged. The
vmapped reverse call returns to the existing median autograd graph and removes
22,946,492 allocated bytes and 515 allocation events in this case. Timings vary
on the shared runner, including slower inference controls, and establish no
general latency guarantee.

The base source SHA-256 in `base.json` is
`ea3674377a02c8fea4029cb7efb44285ab51ac2733eca8ac535eb81669ef9e1a`.
The measured branch source SHA-256 is
`85cb9e2e37c1b0b6744f6ee31aafaad69e7269aec09ce64e546550efb006a98d`.
The benchmark was run with the implementation uncommitted on local commit
`fe4080d`; two explanatory allocation-comment lines were added afterward. The
executable implementation is unchanged in local follow-up commit `b932ab8`.
The JSON records retain the interpreter, imported module, versions and times.

Run from each checkout root so imports resolve to that checkout:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  <python-with-torch> - result.json < median_vmap_bench.py
```

The native CPU platform is x86_64, Python 3.12.14, PyTorch 2.15.0+cpu and
kornia_rs 0.2.0. Accelerator performance is unmeasured. This evidence directory
contains benchmark evidence only; the PR description records the separately
run regression and conformance checks and their actual completion status.
