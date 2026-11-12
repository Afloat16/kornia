DeDoDe keeps its keypoint coordinates in float32 when PyTorch's default dtype is float64, so `forward` can sample descriptors without a grid dtype mismatch.
