from params_preproc import *
# Extracted features with diff because MFCCs are equal for the 2 signals. On the diff they are non-zero, so basically MFCC(diff) != diff(MFCCs)

# Feature extractor
# todo here will be a subfolder structure for each drum channel
extracted_features_folder = r"../data/features-latest"

feature_dict = {'sr': sample_rate, 'n_fft': 4096*4, 'n_mfcc': 26, 'hop_length': 512, 'margin': 3.0, 'n_lvls':5,'wavelet_type':'db1'}
# feature_list = ['mfcc'] # ['spect', 'mel_spect']
# feature_list = ['fft'] # ['spect', 'mel_spect']
feature_list = ['cepstrum'] # ['spect', 'mel_spect']
variance_type = 'smad'  #[string], type of variance, either 'var' or 'smad'
raw_features = True #[bool] if True, skips mean and var extraction from the audio features in the feature list
keep_feature_dims = True #[bool] if True, do not reduce individual features' dimensions to 1D shape. Only useful if raw_features is True

param_pre_diff = True # this parameter specifies if the difference is made before or after the features are extracted.
                    # if True, the features are extracted from the diff. if False, the diff is made from the features.