"""Run the repository's selected conformance suites with bounded CPU threads."""

import os
import sys

if os.environ.get("KORNIA_CONFORMANCE_DIAGNOSTIC") == "1":
    import faulthandler

    trace_file = open("../contrib-validation/conformance-stack.log", "w")
    faulthandler.dump_traceback_later(20, repeat=True, file=trace_file)

# ORT's runtime API can run after its initialization event. Its documented
# process-wide opt-out must be set before any dependency can import ORT.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import torch
import pytest

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

mode = sys.argv[1]
packages = "geometry,augmentation,filters,enhance,losses,metrics,io,image,core,feature"
if mode == "onnx":
    import onnxruntime as ort

    ort.disable_telemetry_events()
    original_session = ort.InferenceSession

    class SingleThreadSession(original_session):
        def __init__(self, path_or_bytes, sess_options=None, *args, **kwargs):
            if sess_options is None:
                sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            super().__init__(path_or_bytes, sess_options, *args, **kwargs)

    # The unmodified survey harness constructs a session without options. Set
    # documented CPU thread options here to avoid a host-sized pool per case.
    ort.InferenceSession = SingleThreadSession
    target = "tests/onnx/test_conformance.py"
elif mode == "compile":
    target = "tests/test_inductor_conformance.py"
else:
    raise ValueError(mode)

print("Torch", torch.__version__, "threads", torch.get_num_threads(), "ORT_DISABLE_TELEMETRY=1", flush=True)
raise SystemExit(pytest.main([target, f"--{mode}-conformance={packages}", "--device=cpu", "--dtype=float32", "--timeout=600", "-q", *sys.argv[2:]]))
