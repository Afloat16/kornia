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

from kornia.contrib import KMeans

from testing.base import BaseTester


class TestKMeansDistances(BaseTester):
    @pytest.mark.parametrize("center_dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("strided", [False, True])
    def test_predict_ties_and_layout(self, device, dtype, center_dtype, strided):
        x = torch.tensor([[-2, 0], [0, 0], [1, 0], [2, 0], [4, 0], [1, 3]], device=device, dtype=dtype)
        centers = torch.tensor([[-1, 0], [3, 0], [1, 3]], device=device, dtype=center_dtype)
        if strided:
            x = x.repeat_interleave(2, dim=1)[:, ::2]
            centers = centers.repeat_interleave(2, dim=1)[:, ::2]
        model = KMeans(3, centers)
        # (1, 0) is exactly two units from the first two centers and must take index zero.
        self.assert_close(model.predict(x), torch.tensor([0, 0, 0, 1, 1, 2], device=device))
        assert model.predict(x[:0]).shape == (0,)

    def test_predict_near_centers_at_common_offset(self, device, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("the quarter-pixel differences at this offset are not all representable in half precision")
        # At this offset, subtracting squared norms rounds nearby distances to zero in float32.
        # The direct distance retains the quarter-pixel separation even with more than 25 centers.
        centers = (1024 + torch.arange(32, device=device, dtype=dtype) * 0.25)[:, None].expand(-1, 64)
        expected = torch.tensor([3, 17, 28], device=device)
        x = centers[expected] + 0.0625
        self.assert_close(KMeans(32, centers).predict(x), expected)

    def test_fit_keeps_mean_gradients(self, device, dtype):
        x = torch.tensor([[-2, 0], [0, 0], [3, 0], [5, 0]], device=device, dtype=dtype, requires_grad=True)
        centers = torch.tensor([[-1, 0], [4, 0]], device=device, dtype=dtype)
        model = KMeans(2, centers, tolerance=None, max_iterations=2)
        model.fit(x)
        self.assert_close(model.cluster_centers, centers)
        self.assert_close(model.cluster_assignments, torch.tensor([0, 0, 1, 1], device=device))
        model.cluster_centers.sum().backward()
        self.assert_close(x.grad, torch.full_like(x, 0.5))

    @pytest.mark.parametrize("mode", ["forward_ad", "func_jvp"])
    def test_fit_keeps_forward_derivatives(self, device, dtype, mode):
        x = torch.tensor([[-2, 0], [0, 0], [3, 0], [5, 0]], device=device, dtype=dtype)
        centers = torch.tensor([[-1, 0], [4, 0]], device=device, dtype=dtype)
        model = KMeans(2, centers, tolerance=None, max_iterations=1)

        def fit(data):
            model.fit(data)
            return model.cluster_centers

        if mode == "forward_ad":
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(x, torch.ones_like(x))
                primal, tangent = torch.autograd.forward_ad.unpack_dual(fit(dual))
        else:
            primal, tangent = torch.func.jvp(fit, (x,), (torch.ones_like(x),))
        self.assert_close(primal, centers)
        self.assert_close(tangent, torch.ones_like(centers))

    def test_vmap_predict(self, device, dtype):
        centers = torch.tensor([[-1, 0], [3, 0]], device=device, dtype=dtype)
        x = torch.tensor([[[-2, 0], [4, 0]], [[2, 0], [1, 0]]], device=device, dtype=dtype)
        self.assert_close(torch.vmap(KMeans(2, centers).predict)(x), torch.tensor([[0, 1], [1, 0]], device=device))

    def test_dynamo_predict(self, device, dtype, torch_optimizer):
        centers = torch.tensor([[-1, 0], [3, 0]], device=device, dtype=dtype)
        model = KMeans(2, centers)
        predict = torch_optimizer(model.predict)
        for x, expected in [([[-2, 0], [1, 0], [4, 0]], [0, 0, 1]), ([[2, 0]], [1])]:
            self.assert_close(predict(centers.new_tensor(x)), torch.tensor(expected, device=device))
