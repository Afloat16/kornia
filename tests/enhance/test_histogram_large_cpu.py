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

import math

import pytest
import torch

from kornia.enhance import image_histogram2d

from testing.base import BaseTester


class TestLargeImageHistograms(BaseTester):
    @pytest.mark.parametrize("shape", [(128, 128), (2, 64, 128), (2, 2, 64, 64)])
    @pytest.mark.parametrize("kernel", ["triangular", "uniform", "epanechnikov", "gaussian"])
    @pytest.mark.parametrize("return_pdf", [False, True])
    def test_repeated_explicit_centers(self, shape, kernel, return_pdf, device, dtype):
        image = torch.full(shape, 0.5, device=device, dtype=dtype)
        centers = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device, dtype=dtype).repeat(64)
        hist, pdf = image_histogram2d(image, centers=centers, bandwidth=0.25, kernel=kernel, return_pdf=return_pdf)
        # Every pixel is 0.5. At the five centers the normalized distances are 2, 1, 0, 1, 2.
        # Gaussian literals are exp(-2), exp(-0.5), 1; compact kernels vanish at or outside their support.
        if kernel == "gaussian":
            weights = [0.1353352832366127, 0.6065306597126334, 1.0, 0.6065306597126334, 0.1353352832366127]
        elif kernel == "uniform":
            weights = [0.0, 1.0, 1.0, 1.0, 0.0]
        else:
            weights = [0.0, 0.0, 1.0, 0.0, 0.0]
        expected = torch.tensor(weights, device=device, dtype=hist.dtype).repeat(64) * math.prod(shape[-2:])
        expected = expected.expand(*shape[:-2], -1)
        self.assert_close(hist, expected)
        expected_pdf = expected / (expected.sum(-1, keepdim=True) + 1e-10) if return_pdf else torch.zeros_like(expected)
        self.assert_close(pdf, expected_pdf)

    @pytest.mark.parametrize("layout", ["contiguous", "transposed", "channels_last"])
    def test_automatic_bin_boundaries(self, layout, device, dtype):
        image = torch.full((1, 2, 65, 129), 0.5, device=device, dtype=dtype)
        if layout == "transposed":
            image = image.transpose(-1, -2)
        elif layout == "channels_last":
            image = image.contiguous(memory_format=torch.channels_last)
        hist, pdf = image_histogram2d(image, 0.0, 1.0, n_bins=256, return_pdf=True)
        # 0.5 lies halfway between automatic centers 127 and 128; each receives half of the 65*129 pixels.
        expected = torch.zeros((1, 2, 256), device=device, dtype=hist.dtype)
        expected[..., 127:129] = 65 * 129 / 2
        self.assert_close(hist, expected, atol=0, rtol=0)
        expected_pdf = torch.zeros_like(expected)
        expected_pdf[..., 127:129] = 0.5
        self.assert_close(pdf, expected_pdf)

    @pytest.mark.parametrize("kernel", ["triangular", "uniform", "epanechnikov", "gaussian"])
    def test_dynamo_large_image(self, kernel, device, dtype, torch_optimizer):
        image = torch.rand((128, 129), device=device, dtype=dtype)

        def operation(value):
            return image_histogram2d(value, 0.0, 1.0, n_bins=256, kernel=kernel, return_pdf=True)

        hist, pdf = operation(image)
        compiled_hist, compiled_pdf = torch_optimizer(operation)(image)
        self.assert_close(compiled_hist, hist)
        self.assert_close(compiled_pdf, pdf)
