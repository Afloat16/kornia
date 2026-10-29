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

from kornia.geometry.transform import Affine

from testing.base import BaseTester


def _parameters(kind, device, dtype):
    values = {
        "angle": [23.0],
        "translation": [[0.3, -0.4]],
        "scale_factor": [[0.9, 1.1]],
        "shear": [[0.1, -0.2]],
    }
    return {
        key: torch.tensor(value, device=device, dtype=dtype)
        for key, value in values.items()
        if kind in (key, "combined")
    }


class TestAffineSharedParameters(BaseTester):
    @pytest.mark.parametrize("kind", ["angle", "translation", "scale_factor", "shear", "combined"])
    @pytest.mark.parametrize("explicit_center", [False, True])
    def test_shared_transform_matches_per_image_6395(self, device, dtype, kind, explicit_center):
        image = torch.linspace(0.0, 1.0, 210, device=device, dtype=dtype).reshape(3, 2, 5, 7)
        parameters = _parameters(kind, device, dtype)
        center = torch.tensor([[3.0, 2.0]], device=device, dtype=dtype) if explicit_center else None
        transform = Affine(**parameters, center=center)
        expected = torch.cat([transform(sample[None]) for sample in image])
        actual = transform(image)
        assert actual.shape == image.shape
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("kind", ["angle", "translation", "scale_factor", "shear", "combined"])
    def test_shared_transform_gradients_match_explicit_center_6395(self, device, dtype, kind):
        image = torch.linspace(0.0, 1.0, 210, device=device, dtype=dtype).reshape(3, 2, 5, 7).requires_grad_()
        reference_image = image.detach().clone().requires_grad_()
        parameters = {key: value.requires_grad_() for key, value in _parameters(kind, device, dtype).items()}
        reference_parameters = {key: value.detach().clone().requires_grad_() for key, value in parameters.items()}
        actual = Affine(**parameters)(image)
        # The supported explicit-center batch path has the same half-precision
        # gradient reduction order as the implicit-center batch path.
        center = torch.tensor([[3.0, 2.0]], device=device, dtype=dtype)
        expected = Affine(**reference_parameters, center=center)(reference_image)
        weights = torch.linspace(0.1, 0.9, actual.numel(), device=device, dtype=dtype).reshape_as(actual)
        actual_grads = torch.autograd.grad(actual, (image, *parameters.values()), weights)
        expected_grads = torch.autograd.grad(expected, (reference_image, *reference_parameters.values()), weights)
        self.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            assert torch.isfinite(actual_grad).all()
            self.assert_close(actual_grad, expected_grad)

    @pytest.mark.parametrize("unbatched", [False, True])
    def test_multiple_transforms_of_one_image(self, device, dtype, unbatched):
        image = torch.linspace(0.0, 1.0, 70, device=device, dtype=dtype).reshape(1, 2, 5, 7)
        angles = torch.tensor([10.0, 20.0, 30.0], device=device, dtype=dtype)
        expected = torch.cat([Affine(angle=angle[None])(image) for angle in angles])
        actual = Affine(angle=angles)(image[0] if unbatched else image)
        self.assert_close(actual, expected)

    def test_empty_image_batch(self, device, dtype):
        image = torch.empty(0, 2, 5, 7, device=device, dtype=dtype)
        angle = torch.tensor([23.0], device=device, dtype=dtype)
        actual = Affine(angle=angle)(image)
        self.assert_close(actual, image)
