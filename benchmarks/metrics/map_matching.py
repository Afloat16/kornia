# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Compare public mean_average_precision matching cost between revisions.

Each image has ten ground-truth boxes split across two classes. Predictions contain
exact matches, duplicates and a disjoint false positive, with fixed random scores.
This synthetic workload measures evaluation throughput, not model quality. Run the
same command from both checkouts and compare the AP values along with latency.

Example:
    python benchmarks/metrics/map_matching.py --device cpu --json map.json
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from functools import partial
from pathlib import Path

import torch

CHECKOUT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHECKOUT))
sys.path.insert(0, str(CHECKOUT / "benchmarks"))

from common import run_metadata, save_json, time_us, warm_up_cpu  # noqa: E402

import kornia  # noqa: E402


def make_inputs(images: int, device: torch.device) -> tuple[list[torch.Tensor], ...]:
    """Build deterministic float32 boxes, class labels, and scores outside timing."""
    box = torch.tensor([[i / 12, 0.1, i / 12 + 0.05, 0.3] for i in range(10)], device=device)
    labels = torch.arange(10, device=device).remainder(2) + 1
    gt_boxes = [box.clone() for _ in range(images)]
    gt_labels = [labels.clone() for _ in range(images)]
    pred_boxes = [torch.cat((box, box, box.new_tensor([[0.85, 0.6, 0.95, 0.9]]))) for _ in range(images)]
    pred_labels = [torch.cat((labels, labels, labels.new_tensor([1]))) for _ in range(images)]
    pred_scores = list(torch.rand(images, 21, device=device).unbind())
    return pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--images", default="1,64,256")
    parser.add_argument("--min-run-time", type=float, default=6.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(73)
    device = torch.device(args.device)
    sync = torch.mps.synchronize if device.type == "mps" else None
    warm_up_cpu()
    results = []
    for images in map(int, args.images.split(",")):
        inputs = make_inputs(images, device)
        run = partial(kornia.metrics.mean_average_precision, *inputs, 3)
        value, per_class = run()
        median, iqr = time_us(run, min_run_time=args.min_run_time, sync=sync)
        row = {
            "images": images,
            "ground_truth": 10 * images,
            "detections": 21 * images,
            "median_us": median,
            "iqr_us": iqr,
            "map": value.item(),
            "per_class": per_class,
        }
        results.append(row)
        print(row, flush=True)
    metadata = run_metadata(device)
    metadata["python_executable"] = sys.executable
    metadata["kornia_file"] = kornia.__file__
    metadata["source_sha256"] = hashlib.sha256(
        (CHECKOUT / "kornia/metrics/mean_average_precision.py").read_bytes()
    ).hexdigest()
    metadata["cpu_affinity"] = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    metadata["seed"] = 73
    metadata["min_run_time"] = args.min_run_time
    print(metadata)
    if args.json is not None:
        save_json(args.json, metadata, results)


if __name__ == "__main__":
    main()
