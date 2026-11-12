Fix EfficientViT's `LayerNorm2d` forward pass when constructed with `bias=False`, preserving the affine scale and input gradients.
