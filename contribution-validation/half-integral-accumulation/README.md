# Half-precision multi-axis integral validation

Source: Afloat16/kornia commit `12fd5c57333ae039a989ffcf950cb32096d6ed80` (tree `f89a26abde806c4c2e55ced1055f0a6ff38e907f`), based on upstream `154cacefcc644c67145834840192cd1816955e20`.

## Defect and expected results

For epsilon equal to the dtype spacing at 1, N=2/epsilon+1 rows of `[1, -(1-epsilon)]` have a final second-column sum `N*epsilon = 2+epsilon`, which rounds to 2 in float16 and bfloat16. The first axis in the original implementation rounds its prefixes before the second axis cancels them, returning 1. Reversing the axes is a passing control. The same cancellation occurs in backward using two rows of weights `[1, -(1-epsilon)]` across N columns, even though the original input is all zero.

A 256x256x2 float16 tensor repeating `[1,-1]` has a positive prefix 65536, outside float16, but its last paired prefix sums to exactly zero. Original multi-axis accumulation returns `[inf,nan]`; the correction returns `[inf,0]`. All input values and gradient weights are within [-1,1]. These are synthetic proofs, not claimed observations from user applications.

The correction casts multiple-axis half inputs to float32 before any cumulative sum and casts the final output back. Single-axis and other dtypes retain their existing path. Explicit input conversion also keeps the entire backward cumulative chain in float32; the intermediate experiment using only cumsum's dtype argument did not fix both gradient cases.

## Results

- Real baseline: 5 failed, 2 passing controls. A preceding marker-setup collection error is separately retained and is not counted as a code regression.
- Final half functional tests: 55 passed (1 inapplicable bfloat16-only exclusion); standard float32/float64 functional tests: 48 passed (10 inapplicable half cases deselected).
- Half Inductor forward/backward regression: 2 passed.
- Official promised-package selector: `augmentation,enhance` for both conformance suites.
- Complete selected ONNX conformance: 199 passed in 177.07s.
- Complete selected compile conformance: 105 passed in 43.54s.
- Additional float16 new-branch checks: legacy and dynamo ONNX exports pass checker and ONNX Runtime, output float16 with final prefix [2048,2].
- 4 doctests, 12 full pre-commit hooks, ty, Towncrier draft rendering and git diff --check passed.

CPU only: Python 3.12.14, torch 2.15.0+cpu, ONNX 1.23.2, ONNX Runtime 1.32.0. No CUDA/MPS run is claimed. Pixi was unavailable; the equivalent commands use the explicit prepared Python environment. The conformance runner only bounds CPU/ORT thread pools and disables telemetry; it retains the repository's unmodified tests and assertions.

## Cost and quality

The same public-API harness ran from the base and patched checkout roots. Each run verifies kornia.__file__, uses one Torch thread, `benchmarks.common.warm_up_cpu`, `time_us` (0.15s blocked_autorange median/IQR) and `run_metadata`. All 21 rows, including slower cases, are retained. Allocation is cumulative positive CPU profiler allocation traffic, not peak memory or RSS. Nonfinite scalar JSON summaries are null; the explicit counters retain their meaning.

For 1x3x128x128 inputs, two-axis half allocation traffic increased 196608 -> 688128 bytes (3.5x); float16 median was 113.052 -> 127.857us and bfloat16 136.605 -> 264.505us on this host. Float16 maximum finite error versus float64 then cast-back decreased .0625 -> .000244140625, and bfloat16 .5 -> 0. Narrow cancellation and volume cases are slower too. All 14 control rows retained their output bytes; single-axis and non-half rows retained allocation counts/bytes. Timing differences in unchanged paths are measurement variability, not an optimization claim. Detailed median/IQR values and source hashes are in the JSON files.

This is a correctness tradeoff, not a speed improvement. Manual float32 conversion is already a supported workaround; the change applies that working precision within the existing APIs and their module forms while preserving caller/output dtype. It does not add a second integration algorithm. Float32 working precision does not guarantee exact results for arbitrary inputs; final sums outside the output dtype range still overflow.

## Reproduction

Run each Python probe as stdin from the intended checkout root with an explicit interpreter. Both functional precision profiles are separate pytest invocations as TESTING.md requires. Inductor uses KORNIA_TEST_OPTIMIZER=inductor. The stored conformance runner receives KORNIA_CONFORMANCE_PACKAGES=augmentation,enhance; it uses the repository's normal --timeout=600 per-case gate and one thread. The benchmark and additional ONNX probe require their existing optional dependencies. No benchmark data or logs are in the upstream library diff.
