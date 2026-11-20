# Polygon-list coordinate staging: public API comparison

This evidence uses upstream Kornia `7ecfbb392285e03f73b4dd4cce62fc3fb8349c33` and source patch `2b4a6811ba743d88225dc15b17b42ba61294672a` on `Afloat16/kornia` (`fix/polygon-list-coordinate-promotion-20261120`). Both runs print and record `sys.executable`, `kornia.__file__`, Git identity, and the SHA-256 of the imported `kornia/image/draw.py`. The branch was measured before its local commit, so its metadata correctly says `7ecfbb3-dirty`; the final code adds only a measured-cost comment to the executed production patch.

Run from each respective checkout root with the same explicit interpreter, passing the harness on stdin:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python - result.json < polygon_list_bench.py
```

The public `draw_convex_polygon` operation draws eight polygons into preallocated float32 `(8, 1, 32, 32)` images. Lists alternate four/five vertices; the tensor control has four vertices. All vertices describe the same rectangle except the explicitly affected mixed case, whose later float32 rectangles start at x=2.5. The latter must fill columns 3..28 (754 pixels), while integer rectangles start at x=2 (783 pixels). The baseline incorrectly fills 783 pixels in every row of that batch; the patch produces the independently expected alternating 783/754 counts.

The standard `benchmarks.common` CPU warm-up and blocked autorange helpers are used, with 0.2-second samples, one CPU thread and a fixed seed. Memory totals are sums of positive PyTorch allocation events from one public call, not peak memory or RSS. Full metadata, IQRs, output hashes and counts are retained in the JSON files.

| Case | Positive bytes: base → patch | Events: base → patch | Median microseconds: base → patch | Output hash |
|---|---:|---:|---:|---|
| uniform_float32 | 111,448 → 111,448 | 43 → 43 | 254.75 → 329.11 | same |
| uniform_float16 | 111,528 → 111,368 | 44 → 43 | 262.71 → 257.53 | same |
| uniform_int64 | 112,248 → 111,608 | 44 → 43 | 249.44 → 274.99 | same |
| uniform_float64 | 151,104 → 151,104 | 43 → 43 | 259.50 → 270.27 | same |
| mixed_int64_float32 | 112,248 → 111,608 | 44 → 43 | 261.67 → 558.43 | changed to the expected pixel counts |
| tensor_control | 101,528 → 101,528 | 36 → 36 | 172.20 → 180.22 | same |

The byte difference is the eliminated narrow staging buffer; it is small for these small polygons. The purpose of the patch is correct coordinates and order-independent dtype selection. Shared-runner timing is noisy: the mixed-case patch IQR is about 1125 microseconds, and float32 IQR about 208 microseconds. All samples, including the slower patch measurements, are retained. These measurements establish neither a general speedup nor a reliable latency regression. Accelerator execution is not measured here.
