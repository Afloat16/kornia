# Keypoint index-update copy validation

This evidence accompanies the input-preserving `Keypoints.index_put` optimization and its inherited `VideoKeypoints` path. It is a performance change: the functional regression tests also pass against the unchanged implementation. The measured defect is a second complete coordinate allocation/copy that is immediately discarded.

## Source identity

- Base: `7ecfbb392285e03f73b4dd4cce62fc3fb8349c33`.
- Patched public commit: `ee76137ddb1de2bcad3d991c11f6f2733457e2f1`.
- Patched tree: `1a385698fc88315e2a96262cfb4399b76788f668`.
- Equivalent local commit: `e611627b3d22ce062527f348203ea30f14f40d2f`.
- Newer main `154cacefcc644c67145834840192cd1816955e20` only changes comments/docstrings in `geometry/line.py`. `git merge-tree --write-tree` succeeds without overlap; no executed code differs from the validated base.

The change clones the container once and updates that clone, retaining the existing input-preserving contract. Using `inplace=True` instead would change that contract. No new API, data-dependent branch, cache, or operator specialization is introduced. `VideoKeypoints.clone()` continues to preserve the temporal metadata.

## Reproducible measurements

Use `round2_keypoints_copy_bench.py` as stdin from each checkout root so imports select that checkout, with its Python environment containing the repository dependencies:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python - before.json < round2_keypoints_copy_bench.py
```

The runner prints the interpreter and imported `kornia.__file__`. Inputs are preallocated, CPU threads are one, and the repository's `benchmarks.common.warm_up_cpu` and `time_us` are used (three-second warmup and 0.15-second minimum measurement). Allocation figures sum positive `torch.profiler` event `self_cpu_memory_usage`, with event counts. These are cumulative allocation traffic, not peak memory, retained memory, or process RSS. Input and output ownership/content checks occur outside the measured call. All twelve base/patch output SHA-256 digests match.

| Input, float32 | Default-call allocation bytes, base → patch | Allocation events | Median microseconds, base → patch | IQR microseconds, base → patch |
| --- | ---: | --- | ---: | ---: |
| Pose `(17, 2)` | 272 → 136 | 2 → 1 | 5.519 → 4.644 | 0.449 → 0.923 |
| Pose batch `(32, 17, 2)` | 8,704 → 4,352 | 2 → 1 | 5.657 → 4.650 | 0.448 → 0.608 |
| Features `(32, 2048, 2)` | 1,048,576 → 524,288 | 2 → 1 | 35.280 → 22.597 | 3.322 → 0.744 |
| Video `(2, 16, 2048, 2)` | 1,048,576 → 524,288 | 2 → 1 | 35.461 → 22.803 | 0.980 → 3.267 |
| Strided `(32, 2048, 2)` | 1,048,576 → 524,288 | 2 → 1 | 36.969 → 21.060 | 2.804 → 1.056 |
| Empty `(0, 2)` | 0 → 0 | 0 → 0 | 4.630 → 3.661 | 0.322 → 0.185 |

The corresponding `inplace=True` controls allocate zero tensor bytes/events in both versions. Their timing samples include modest increases as well as decreases; all values and IQRs are retained in the two JSON results and comparison. The memory reduction follows from removing one full clone. CPU latency measurements on this shared runner do not establish an accelerator speedup or universal latency guarantee.

## Checks

Python 3.12.14, PyTorch 2.15.0+cpu, ONNX Runtime 1.32.0, ONNX 1.23.2, one CPU/ORT/Inductor thread. Repository tasks were invoked directly in an isolated environment because Pixi was unavailable. Export/runtime assertions were unchanged.

- Before changing production: 36 behavioral checks passed, 233 deselected.
- Complete keypoints functional file for float32/float64: 269 passed, 9 warnings, 0.70 seconds.
- Focused update/gradient cases across float16/bfloat16/float32/float64: 72 passed, 455 deselected, 1 warning, 0.52 seconds.
- Official changed-path selection: `geometry,augmentation` for both conformance suites.
- Complete selected ONNX conformance: 903 passed, 5 warnings, 735.74 seconds.
- Complete selected compile conformance with Inductor: 451 passed, 2 warnings, 156.19 seconds.
- All 12 pre-commit hooks, `ty check kornia`, Towncrier preview, and `git diff --check` passed.
- Independent read-only review checked aliasing, original-input preservation, strided storage, gradients, and video metadata. It found no blocking issue and did not run tests.

The new behavior tests cover both tensor/container values, normal/video containers, contiguous/strided input, update-in-place/returned-copy behavior, and gradients to original coordinates and replacement values. Accelerator and older supported PyTorch execution remain for CI.

The included logs retain warnings and raw benchmark samples. ONNX logs contain a buffered SE2 graph dump after the pytest summary; the process exited successfully. No unavailable or skipped suite is counted as passed.

## History boundary

All-author searches for Keypoints/index_put/clone/copy/allocation and 116 contributor fork branch names found no equivalent change. #5069 concerns keypoint coordinate conventions, #6376 video indexing, and #5840 noncontiguous video reshaping. The corresponding `Boxes.index_put` already uses the clone-once pattern and is unchanged. This contribution addresses a different allocation site from the existing geometry/drawing PRs.
