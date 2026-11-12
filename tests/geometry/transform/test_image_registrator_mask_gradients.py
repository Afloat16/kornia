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

from kornia.geometry.transform import Homography, HomographyWarper, ImageRegistrator

from testing.base import BaseTester


class TestImageRegistratorMaskGradients(BaseTester):
    def test_identity_loss_and_image_gradients(self, device, dtype):
        src = torch.tensor([[[[0.25, 0.5], [0.5, 0.75]]]], device=device, dtype=dtype, requires_grad=True)
        dst = torch.zeros_like(src, requires_grad=True)
        matrix = torch.eye(3, device=device, dtype=dtype)[None].requires_grad_()
        loss = ImageRegistrator().get_single_level_loss(src, dst, matrix)
        self.assert_close(loss, src.new_tensor(0.5))
        loss.backward()
        self.assert_close(src.grad, torch.full_like(src, 0.25))
        self.assert_close(dst.grad, torch.full_like(dst, -0.25))
        assert torch.isfinite(matrix.grad).all()

    def test_outside_pixels_do_not_contribute(self, device, dtype):
        src = torch.full((1, 1, 2, 4), 0.5, device=device, dtype=dtype, requires_grad=True)
        dst = src.detach().new_tensor([[[[0.25, 0.25, 0.25, 1], [0.25, 0.25, 0.25, 1]]]]).requires_grad_()
        # Destination-to-source normalized x translation 0.5 samples one pixel to the right.
        matrix = src.new_tensor([[[1, 0, 0.5], [0, 1, 0], [0, 0, 1]]]).requires_grad_()
        loss = ImageRegistrator().get_single_level_loss(src, dst, matrix)
        self.assert_close(loss, src.new_tensor(0.25))
        loss.backward()
        expected_src = src.new_tensor([[[[0, 1 / 6, 1 / 6, 1 / 6], [0, 1 / 6, 1 / 6, 1 / 6]]]])
        expected_dst = -src.new_tensor([[[[1 / 6, 1 / 6, 1 / 6, 0], [1 / 6, 1 / 6, 1 / 6, 0]]]])
        self.assert_close(src.grad, expected_src)
        self.assert_close(dst.grad, expected_dst)
        assert torch.isfinite(matrix.grad).all()

    def test_gradcheck(self, device):
        # The scale keeps every sample inside the image, away from bilinear knots and mask thresholds.
        src = torch.tensor([[[[0.25, 0.5], [0.5, 0.75]]]], device=device, dtype=torch.float64, requires_grad=True)
        dst = torch.zeros_like(src, requires_grad=True)
        matrix = src.new_tensor([[[0.8, 0.03, 0.02], [-0.02, 0.75, -0.04], [0.01, -0.01, 1]]]).requires_grad_()
        self.gradcheck(ImageRegistrator().get_single_level_loss, (src, dst, matrix))

    def test_outer_no_grad_is_preserved(self, device, dtype):
        src = torch.ones(1, 1, 2, 2, device=device, dtype=dtype, requires_grad=True)
        matrix = torch.eye(3, device=device, dtype=dtype)[None].requires_grad_()
        with torch.no_grad():
            loss = ImageRegistrator().get_single_level_loss(src, torch.zeros_like(src), matrix)
            assert not loss.requires_grad
            assert not torch.is_grad_enabled()
        assert torch.is_grad_enabled()

    def test_custom_warper_keeps_grad_mode(self, device, dtype):
        class GradDependentWarper(HomographyWarper):
            def forward(self, patch_src, src_homo_dst=None):
                # A custom warper may differentiate a field internally to obtain its sampling coordinates.
                scale = torch.autograd.grad(src_homo_dst.square().sum(), src_homo_dst, create_graph=True)[0]
                return super().forward(patch_src, src_homo_dst) * (scale[:, 0, 0] / 2).view(-1, 1, 1, 1)

        src = torch.ones(1, 1, 2, 2, device=device, dtype=dtype, requires_grad=True)
        matrix = torch.eye(3, device=device, dtype=dtype)[None].requires_grad_()
        reg = ImageRegistrator(Homography(), warper=GradDependentWarper)
        loss = reg.get_single_level_loss(src, torch.zeros_like(src), matrix)
        self.assert_close(loss, src.new_tensor(1))
        loss.backward()
        assert torch.isfinite(matrix.grad).all()

    @pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
    def test_mixed_transform_under_autocast(self, device, autocast_dtype):
        if device.type != "cpu":
            pytest.skip("this test pins CPU autocast reconciliation")
        src = torch.full((1, 1, 2, 4), 0.5, device=device, requires_grad=True)
        dst = torch.full_like(src, 0.25)
        matrix = src.new_tensor([[[1, 0, 0.5], [0, 1, 0], [0, 0, 1]]]).requires_grad_()
        with torch.autocast("cpu", dtype=autocast_dtype):
            mixed_matrix = matrix @ torch.eye(3, device=device)
            loss = ImageRegistrator().get_single_level_loss(src, dst, mixed_matrix)
        self.assert_close(loss, src.new_tensor(0.25))
        loss.backward()
        assert torch.isfinite(matrix.grad).all()

    def test_dynamo_loss(self, device, dtype, torch_optimizer):
        src = torch.full((1, 1, 2, 4), 0.5, device=device, dtype=dtype)
        dst = torch.full_like(src, 0.25)
        matrix = src.new_tensor([[[1, 0, 0.5], [0, 1, 0], [0, 0, 1]]])
        compiled = torch_optimizer(ImageRegistrator().get_single_level_loss)
        self.assert_close(compiled(src, dst, matrix), src.new_tensor(0.25))
