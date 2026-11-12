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

import pytest
import torch

from kornia.filters import median_blur

from testing.base import BaseTester


class TestMedianCPUWindows(BaseTester):
    # Independent 3x7 median oracle: for each output, pad the 4x5 literals according to the named border,
    # sort its 21 samples, and select sample 10. The gradient counts how often each original pixel is selected.
    @pytest.mark.parametrize(
        "border,expected,gradient",
        [
            (
                "constant",
                [0, 0, 0, 0, 0, 2, 5, 5, 5, 2, 4, 8, 8, 8, 4, 0, 0, 0, 0, 0],
                [0, 0, 2, 0, 3, 0, 0, 0, 0, 0, 3, 0, 2, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                "reflect",
                [9, 11, 9, 11, 9, 11, 9, 11, 9, 12, 10, 11, 10, 11, 10, 12, 10, 12, 10, 12],
                [0, 0, 0, 0, 0, 6, 5, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5, 0],
            ),
            (
                "replicate",
                [3, 5, 5, 5, 5, 8, 8, 8, 8, 7, 11, 11, 11, 12, 13, 15, 13, 13, 13, 13],
                [1, 0, 0, 0, 4, 3, 0, 0, 0, 0, 4, 0, 0, 1, 1, 0, 0, 1, 0, 5],
            ),
            (
                "circular",
                [11, 11, 11, 11, 11, 9, 9, 8, 9, 9, 11, 11, 11, 11, 11, 10, 10, 8, 10, 10],
                [0, 0, 0, 0, 0, 10, 4, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 4, 0],
            ),
        ],
    )
    @pytest.mark.parametrize("layout", ["contiguous", "strided", "channels_last"])
    def test_rectangular_values_and_gradients(self, border, expected, gradient, layout, device, dtype):
        values = (
            torch.tensor(
                [3, 14, 2, 17, 5, 11, 9, 19, 1, 16, 8, 20, 4, 12, 7, 18, 6, 15, 10, 13], device=device, dtype=dtype
            ).reshape(1, 1, 4, 5)
            / 32
        )
        scales = torch.tensor([1, 2, 3, 4], device=device, dtype=dtype).reshape(2, 2, 1, 1) / 4
        values = values * scales
        if layout == "strided":
            values = values.transpose(-1, -2).contiguous().transpose(-1, -2)
        elif layout == "channels_last":
            values = values.contiguous(memory_format=torch.channels_last)
        values.requires_grad_()
        result = median_blur(values, (3, 7), border)
        oracle = torch.tensor(expected, device=device, dtype=dtype).reshape(1, 1, 4, 5) / 32 * scales
        self.assert_close(result, oracle, atol=0, rtol=0)
        grad = torch.autograd.grad(result.sum(), values)[0]
        grad_oracle = torch.tensor(gradient, device=device, dtype=dtype).reshape(1, 1, 4, 5).expand_as(values)
        self.assert_close(grad, grad_oracle, atol=0, rtol=0)
        assert result.is_contiguous()

    @pytest.mark.parametrize("border", ["constant", "reflect", "replicate", "circular"])
    @pytest.mark.parametrize("kernel_size", [7, (3, 7)])
    def test_cpu_negative_zero_matches_convolution(self, border, kernel_size, device, dtype):
        if device.type != "cpu":
            pytest.skip("CPU convolution's signed-zero convention")
        values = torch.full((1, 2, 4, 5), -0.0, device=device, dtype=dtype)
        result = median_blur(values, kernel_size, border)
        self.assert_close(result, torch.zeros_like(result), atol=0, rtol=0)
        assert not result.signbit().any()
