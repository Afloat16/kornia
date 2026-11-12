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

from kornia.contrib import FaceDetector
from kornia.models.yunet import YuNet

from testing.base import BaseTester


class TestFaceDetectionBatchPriors(BaseTester):
    @pytest.fixture
    def detector(self, device, dtype, monkeypatch):
        state = YuNet("test", pretrained=False).state_dict()
        monkeypatch.setattr("kornia.models.yunet.model.load_state_dict_from_url", lambda *args, **kwargs: state)
        return FaceDetector().to(device, dtype)

    def test_distinct_batch_geometry(self, detector, device, dtype):
        # 64x64 has 235 YuNet anchors. The first and fourth are 10px squares
        # centered at (4, 4) and (12, 4), with all five zero-offset landmarks at the center.
        loc = torch.zeros(3, 235, 14, device=device, dtype=dtype)
        conf = torch.zeros(3, 235, 2, device=device, dtype=dtype)
        iou = torch.ones(3, 235, 1, device=device, dtype=dtype)
        conf[0, 0, 1] = 0.81
        conf[1, 3, 1] = 0.64
        data = {"loc": loc, "conf": conf, "iou": iou}
        before = {name: tensor.clone() for name, tensor in data.items()}
        actual = detector.postprocess(data, 64, 64)
        expected = [
            [-1, -1, 9, 9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0.9],
            [7, -1, 17, 9, 12, 4, 12, 4, 12, 4, 12, 4, 12, 4, 0.8],
        ]
        assert len(actual) == 3
        for result, row in zip(actual[:2], expected):
            self.assert_close(result, loc.new_tensor([row]))
        assert actual[2].shape == (0, 15)
        for name, tensor in data.items():
            self.assert_close(tensor, before[name])
        assert detector.postprocess({name: tensor[:0] for name, tensor in data.items()}, 64, 64) == []

    @pytest.mark.parametrize("shape", [(64, 96), (96, 80)])
    def test_native_model_batch_matches_individual_postprocessing(self, detector, device, dtype, shape):
        detector.confidence_threshold = 0.3
        detector.top_k = 20
        detector.keep_top_k = 5
        image = torch.rand(3, 3, *shape, device=device, dtype=dtype) * 255
        with torch.no_grad():
            raw = detector.model(detector.preprocess(image))
            actual = detector.postprocess(raw, *shape)
            expected = [
                detector.postprocess({name: tensor[i : i + 1] for name, tensor in raw.items()}, *shape)[0]
                for i in range(3)
            ]
        for result, reference in zip(actual, expected):
            self.assert_close(result, reference)

    def test_jit_batch_and_empty(self, detector, device, dtype):
        detector.confidence_threshold = 0.99
        compiled = torch.jit.script(detector)
        for batch in (2, 0):
            image = torch.rand(batch, 3, 64, 64, device=device, dtype=dtype) * 255
            with torch.no_grad():
                actual, expected = compiled(image), detector(image)
            assert len(actual) == len(expected) == batch
            for result, reference in zip(actual, expected):
                self.assert_close(result, reference)
