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

import struct

import pytest
import torch

from kornia.geometry import save_pointcloud_ply_binary

from testing.base import BaseTester


class TestBinaryPlyBuffer(BaseTester):
    @pytest.mark.parametrize("layout", ["contiguous", "strided", "transposed", "negative"])
    def test_exact_payload(self, tmp_path, device, dtype, layout):
        points = torch.arange(12, device=device, dtype=dtype).reshape(2, 2, 3) / 4
        expected = [value / 4 for value in range(12)]
        if layout == "strided":
            storage = torch.zeros(2, 2, 6, device=device, dtype=dtype)
            storage[..., ::2] = points
            points = storage[..., ::2]
            assert not points.is_contiguous()
        elif layout == "transposed":
            points = points.transpose(0, 1)
            expected = [value / 4 for value in (0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11)]
            assert not points.is_contiguous()
        elif layout == "negative":
            points = torch._neg_view(points)
            expected = [-value for value in expected]
            assert points.is_neg()
        points.requires_grad_()
        before = points.clone()

        path = tmp_path / "buffer.ply"
        save_pointcloud_ply_binary(path, points)
        header, payload = path.read_bytes().split(b"end_header\n", 1)

        assert b"format binary_little_endian 1.0\n" in header
        assert b"element vertex 4\n" in header
        assert payload == struct.pack("<12d", *expected)
        self.assert_close(points, before)
        assert points.grad is None

    def test_float64_precision_and_special_values(self, tmp_path):
        values = [
            5123456.789012,
            0.1234567890123456,
            -0.0,
            float("inf"),
            -float("inf"),
            float("nan"),
        ]
        points = torch.tensor(values, dtype=torch.float64).reshape(2, 3)
        path = tmp_path / "precision.ply"

        save_pointcloud_ply_binary(path, points)

        payload = path.read_bytes().split(b"end_header\n", 1)[1]
        assert payload == struct.pack("<6d", *values)

    def test_empty_payload(self, tmp_path, device, dtype):
        points = torch.empty(2, 0, 3, device=device, dtype=dtype)
        path = tmp_path / "empty.ply"

        save_pointcloud_ply_binary(path, points)

        header, payload = path.read_bytes().split(b"end_header\n", 1)
        assert b"element vertex 0\n" in header
        assert payload == b""
