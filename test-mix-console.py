# import matplotlib.pyplot as plt
# from utils import *
from feature_extractor import *

## import run parameters from the params files
from signal_processor import *
from params_features import *
# exec(open("params_preproc.py", 'r').read())
# exec(open("params_features.py", 'r').read())


# https://pypi.org/project/yodel/
# https://codeandsound.wordpress.com/2014/10/09/parametric-eq-in-python/
# https://github.com/topics/equalizer?l=python

# TODO make another class for the feature extraction because such a class wouldn't necessarily be project specific HERE

# todo add some global dict that contains the normalization values for the parameters -
#  this will be added to the init with setattr and a for loop

# todo also add the file paths for the input and output signals
#  also add the sample rate

# The diff maker and the feature extractors will load the stuff from the 'global' params

# TODO make a (global) param that allows for choosing if diff is feature based or signal based
#  if it's feature based:
#       1) extract features from signal
#       2) diff features
#  if it's signal based:
#       1) the diff maker will be used on the signal
#       2) extract features from the diff

# TODO maybe add a functionality that allows the user to limit the duration of
#  the input signal to like... 2-3-5 min whatever

# input_sig <-> output&metadata


aas = SignalProcessor(sig_path, resample_to=sample_rate, features_folder=extracted_features_folder, processed_signals_root_folder=out_signals_root_folder)
# TODO update the FeatureExtractor class and make it more dynamic. Also update the files in Features_Functions folder


# Pipeline steps params
proc_end_to_end = False
create_training_features = True

# feature_dict = {'sr': sample_rate, 'n_fft': 4096*4, 'n_mfcc': 26, 'hop_length': 512, 'margin': 3.0, 'n_lvls':5,'wavelet_type':'db1'}
# feature_list = ['mfcc'] # ['spect', 'mel_spect']
# variance_type = 'smad'  #[string], type of variance, either 'var' or 'smad'
# raw_features = True #[bool] if True, skips mean and var extraction from the audio features in the feature list
# keep_feature_dims = True #[bool] if True, do not reduce individual features' dimensions to 1D shape. Only useful if raw_features is True

# feats_extractor = FeatureExtractor(feature_list, feature_dict, variance_type,raw_features,keep_feature_dims)
# # print feats_extractor attributes
# proc_end_to_end = False
# create_training_features = True
# param_pre_diff = True # this parameter specifies if the difference is made before or after the features are extracted.
#                     # if True, the features are extracted from the diff. if False, the diff is made from the features.



if proc_end_to_end:
    today = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    shutil.copyfile("params_preproc.py", os.path.join(out_signals_root_folder, f"params-preproc-{today}.txt"))
    # Usage tips: You need to add numbers at the end of every signal processing type, because
    # you can have multiple of the same type such as peak1, peak2, peak3 etc. - always name them with numbers at the end

    # Usage tips: include dbgain 0 if you want to ignore a certain type of filter OR remove it from the below dict.

    # Change this to the number of filters you want to use or None
    # to use all possible combinations of filters, any number of filters.
    no_filters = len(dict_all_filter_settings)

    # aas._create_all_proc_vars_combinations()
    dict_filenames_and_process_variants = aas.create_end_to_end_all_proc_vars_combinations(dict_all_filter_settings,
                                                                                           root_filename="eq_ed",
                                                                                           start_index=0,
                                                                                           end_index=None,
                                                                                           number_of_filters=no_filters)
    aas.process_signal_all_variants(dict_filenames_and_process_variants) # TODO maybe add run date to the metadata or name of the processed signals
    # for d in dict_filenames_and_process_variants:
    #     print("file name in dict_filenames_and_process_variants", d, '-----')
    #     print(len(set(dict_filenames_and_process_variants[d].keys())))
    #     print(dict_filenames_and_process_variants[d])
if create_training_features:
    today = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    shutil.copyfile("params_features.py", os.path.join(extracted_features_folder, f"params-features-{today}.txt"))
    feats_extractor = FeatureExtractor(feature_list, feature_dict, variance_type, raw_features, keep_feature_dims)
    aas.create_features_diff_for_training(obj_feature_extractor=feats_extractor,
                                          processed_audio_folder=out_signals_root_folder, pre_diff=param_pre_diff, process_entire_signal=True)

# aas.process_signal_all_variants(signal_in, {test_fname: dict_filenames_and_process_variants[test_fname]})
# training_data = aas.load_labels_metadata_for_training(out_signals_root_folder)
# path = r'F:\PCON\Disertatie\AutoMixMaster\datasets\diverse-test\white-noise.wav'
