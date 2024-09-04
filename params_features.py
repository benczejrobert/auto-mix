from params_preproc import *
# Extracted features with diff because MFCCs are equal for the 2 signals. On the diff they are non-zero, so basically MFCC(diff) != diff(MFCCs)

# Feature extractor
# todo here will be a subfolder structure for each drum channel
# TODO create a vizualization for a smaller subset of features BEFORE the feature extraction just to see their separability etc.
#  and maybe a PCA analysis as well
#  + to see if the features are extracted correctly


extracted_features_folder = "..\\data\\features-latest"
n_fft = 4096
feature_dict = {'sr': sample_rate, 'n_fft': n_fft, 'n_mfcc': 26, 'hop_length': 512, 'win_length': n_fft, 'margin': 3.0, 'n_lvls':5,'wavelet_type':'db1'}
feature_list = ['mel_spect'] # ['spect', 'mel_spect', 'mfcc']
# feature_list = ['fft'] # ['spect', 'mel_spect']
# feature_list = ['cepstrum'] # ['spect', 'mel_spect']
variance_type = 'smad'  #[string], type of variance, either 'var' or 'smad'
raw_features = True #[bool] if True, skips mean and var extraction from the audio features in the feature list
keep_feature_dims = True #[bool] if True, do not reduce individual features' dimensions to 1D shape. Only useful if raw_features is True
process_entire_signal = False # [bool] if True, only calculate the features for the first n_fft samples of the signal (i.e. 2-D representations will be of shape (n_fft or n_coeffs,1), etc.)
param_pre_diff = False # this parameter specifies if the difference is made before or after the features are extracted.
                    # if True, the features are extracted from the diff. if False, the diff is made from the features.