def fft(self):
    # https://stackoverflow.com/questions/12116830/numpy-slice-of-arbitrary-dimensions
    # or exec(f"b = a[{':,'*no_shapes}index]") and then use b
    fourier = FFT(self.signal, self.n_fft)
    return amplitude(fourier[..., 0:fourier.shape[-1] // 2 + 1])  # get the last index from the dimensions of the array