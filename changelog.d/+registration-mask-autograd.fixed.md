Reduce ImageRegistrator peak memory by avoiding the unused autograd graph for the built-in homography coverage mask, while preserving loss/image/model gradients and custom-warper grad mode.
