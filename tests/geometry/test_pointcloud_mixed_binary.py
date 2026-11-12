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

from kornia.geometry import load_pointcloud_ply_binary

from testing.base import BaseTester


@pytest.mark.device_agnostic
class TestMixedBinaryPointCloud(BaseTester):
    @pytest.mark.parametrize("endian", ["<", ">"])
    @pytest.mark.parametrize("scalar, alias, code", [("float", "float32", "f"), ("double", "float64", "d")])
    @pytest.mark.parametrize(
        "order", [("x", "y", "z", "red", "green", "blue"), ("red", "z", "green", "x", "blue", "y")]
    )
    def test_colored_vertices(self, tmp_path, endian, scalar, alias, code, order):
        # Independently packed colored vertices, including unaligned coordinates and mixed PLY type spellings.
        rows = [
            {"x": 1.25, "y": -2.5, "z": 3.75, "red": 255, "green": 0, "blue": 7},
            {"x": -0.0, "y": float("inf"), "z": float("nan"), "red": 1, "green": 2, "blue": 3},
            {"x": 4.5, "y": -6.0, "z": 0.125, "red": 12, "green": 34, "blue": 56},
        ]
        properties = {name: (alias if name == "y" else scalar) for name in ("x", "y", "z")}
        properties.update(dict.fromkeys(("red", "green", "blue"), "uchar"))
        fmt = "binary_little_endian" if endian == "<" else "binary_big_endian"
        header = f"ply\nformat {fmt} 1.0\nelement camera 1\nproperty float focal\nelement vertex 3\n"
        header += "".join(f"property {properties[name]} {name}\n" for name in order)
        header += "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
        row_format = endian + "".join(code if name in ("x", "y", "z") else "B" for name in order)
        payload = b"".join(struct.pack(row_format, *(row[name] for name in order)) for row in rows)
        filename = tmp_path / "colored.ply"
        filename.write_bytes(header.encode("ascii") + struct.pack(endian + "f", 500) + payload + bytes(13))

        actual = load_pointcloud_ply_binary(filename)
        second = load_pointcloud_ply_binary(filename)
        filename.unlink()
        expected = torch.tensor([[1.25, -2.5, 3.75], [-0.0, float("inf"), float("nan")], [4.5, -6.0, 0.125]])
        assert actual.device.type == "cpu"
        assert actual.dtype == torch.float32
        self.assert_close(actual.isnan(), expected.isnan())
        self.assert_close(actual[~expected.isnan()], expected[~expected.isnan()])
        assert torch.signbit(actual[1, 0])
        actual.zero_()
        self.assert_close(second.isnan(), expected.isnan())
        self.assert_close(second[~expected.isnan()], expected[~expected.isnan()])

    @pytest.mark.parametrize("endian", ["<", ">"])
    def test_mixed_coordinate_types(self, tmp_path, endian):
        fmt = "binary_little_endian" if endian == "<" else "binary_big_endian"
        header = f"ply\nformat {fmt} 1.0\nelement vertex 2\n"
        header += "property float x\nproperty uchar red\nproperty double y\nproperty short z\nend_header\n"
        filename = tmp_path / "mixed.ply"
        filename.write_bytes(
            header.encode("ascii") + struct.pack(endian + "fBdhfBdh", 1.25, 255, -2.5, 3, -4.5, 0, 5.25, -6)
        )
        self.assert_close(load_pointcloud_ply_binary(filename), torch.tensor([[1.25, -2.5, 3], [-4.5, 5.25, -6]]))
