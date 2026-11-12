`image_histogram2d` bounds large eager CPU inference temporaries by processing bins in chunks, preserving each bin's pixel reduction and the existing gradient and graph-capture paths.
