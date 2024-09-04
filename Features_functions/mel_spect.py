def mel_spectrogram(self):
    # TODO testme @20240318 - updated interface
    # if (hasattr(self, 'spect') == False): # spect must be reset after feature was extracted. rather check features list
    if 'spect' not in self.feature_list: # spect must be reset after feature was extracted. rather check features list
        print("spect not in feature_list, calculating it now")
        self.feature_spect(self)
        # spect = amplitude(librosa.stft(self.signal, n_fft=self.n_fft, hop_length=self.hop_length))
        # setattr(self, 'spect', spect)
    mel_spect = librosa.feature.melspectrogram(S=self.spect ** 2, sr=self.sr)
    setattr(self, 'mel_spect', mel_spect)
    return mel_spect #todo what do if no self.spect calculated?

# default mel kwargs
# sr,
#     n_fft,
#     n_mels=128,
#     fmin=0.0,
#     fmax=None,
#     htk=False,
#     norm="slaney",
#     dtype=np.float32,
