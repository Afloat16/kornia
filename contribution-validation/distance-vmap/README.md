# Distance-transform derivative cost probe

Run the Python harness from each Kornia worktree root with an explicit Python environment and `PYTHONPATH=.`. The harness prints and records import provenance, uses `benchmarks.common` for warmup/timing and records positive CPU tensor allocation events through PyTorch's profiler. These are allocation traffic measurements, not peak memory or process RSS.

Base: `7ecfbb392285e03f73b4dd4cce62fc3fb8349c33`. Candidate: `29e91d876f874153b242775368757971f8fa0e1c`. The candidate samples were measured before adding the two-line cost comment beside `_needs_grad`; removing only that comment reproduces the recorded source SHA-256. The executable code is identical.

The input is a float32 tensor of shape `(2, 1, 32, 32)` with source values in `[0.25, 0.75]`. Every source selects itself, so minima are unique. All four output digests and both gradient digests match base exactly. The new tie-gradient regressions separately pin the intended first-source derivative.

Ordinary inference, ordinary reverse differentiation and `vmap` inference allocation traffic match base. `vmap` forward plus reverse differentiation falls from 3,047,804 to 852,348 positive allocation bytes by using the existing detached-argmin/gather derivative path.

Timings were collected on a shared runner and vary in unchanged controls. In particular, ordinary inference's median rose from 143.0 to 207.0 microseconds while vmap inference fell from 450.7 to 362.6. These samples do not establish a latency gain or regression; all timing values and IQRs are retained in the JSONs.
