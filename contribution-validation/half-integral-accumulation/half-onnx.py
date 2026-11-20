"""Exercise the changed half branch through both existing ONNX exporters."""
import os
os.environ['ORT_DISABLE_TELEMETRY'] = '1'
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
import kornia
import onnx
import onnxruntime as ort

torch.set_num_threads(1)
ort.disable_telemetry_events()
print('python', sys.executable, 'kornia', kornia.__file__, flush=True)
assert Path(kornia.__file__).resolve().parent.parent == Path.cwd().resolve()
eps = torch.finfo(torch.float16).eps
length = int(2 / eps) + 1
data = torch.tensor([1., -(1.-eps)], dtype=torch.float16).expand(length, 2)
expected = torch.tensor([length, 2.], dtype=torch.float16)
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
with TemporaryDirectory(dir='../contrib-validation') as work:
    for dynamo in (False, True):
        path = Path(work) / f'integral-{dynamo}.onnx'
        torch.onnx.export(kornia.enhance.IntegralImage(), (data,), path,
                          input_names=['image'], output_names=['output'], dynamo=dynamo, opset_version=18)
        onnx.checker.check_model(path)
        session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        result = torch.from_numpy(session.run(None, {'image': data.numpy()})[0])
        torch.testing.assert_close(result[-1], expected)
        assert result.dtype == torch.float16
        print('dynamo', dynamo, 'dtype', result.dtype, 'last_prefix', result[-1].tolist(), 'PASS', flush=True)
