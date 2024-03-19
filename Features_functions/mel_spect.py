def mel_spectrogram(self):
    # TODO testme @20240318 - updated interface
    if (hasattr(self, 'spect') == False):
        self.spectrogram()
        # spect = amplitude(librosa.stft(self.signal, n_fft=self.n_fft, hop_length=self.hop_length))
        # setattr(self, 'spect', spect)
    mel_spect = librosa.feature.melspectrogram(S=self.spect ** 2, sr=self.sr)
    setattr(self, 'mel_spect', mel_spect)
    return mel_spect
#todo what do if no self.spect calculated?