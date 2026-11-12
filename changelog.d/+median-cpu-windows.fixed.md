`median_blur` extracts CPU eager windows without a dense one-hot convolution, reducing the cost of larger windows and input gradients while preserving the small-window inference path.
