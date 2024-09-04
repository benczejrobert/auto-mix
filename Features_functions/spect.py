def spectrogram(self): # win_len = n_fft by default, hop_len = 25% win_len default, so 75% overlap
    spect = amplitude(librosa.stft(self.signal, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length))
    setattr(self, 'spect', spect) # todo add sr as well
    return spect
