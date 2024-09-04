def mfcc(self):
    # TODO testme @20240318 - updated interface
    # if (hasattr(self, 'mel_spect') == False):
    if 'spect' not in self.feature_list:
        self.feature_mel_spect(self)
        # mel_spect = librosa.feature.melspectrogram(S=self.spect ** 2, sr=self.sr)
        # setattr(self, 'mel_spect', mel_spect)
    retme = librosa.feature.mfcc(S=librosa.power_to_db(self.mel_spect), n_mfcc=self.n_mfcc,
                                 win_length = self.win_length, hop_length=self.hop_length)
    # print(retme.shape)
    return retme #todo what if no self.mel_spect?