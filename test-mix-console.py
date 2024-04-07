# import matplotlib.pyplot as plt
from feature_extractor import *

## import run parameters from the params files
from signal_processor import *
from params_features import *
from split_dataset import *
from k_fold_cross_validation import *
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

# TODO when multiple channels will be added aas will need to have its params reset: sig_root_path, features_folder and
#  processed_signals_root_folder
aas = SignalProcessor(sig_root_path, dict_norm_values=dict_normalization_values, resample_to=sample_rate,
                      features_folder=extracted_features_folder,
                      processed_signals_root_folder=preproc_signals_root_folder)
# TODO update the FeatureExtractor class and make it more dynamic. Also update the files in Features_Functions folder


# Pipeline steps params
proc_end_to_end = False
create_training_features = False
split = True
split_perc_train = 70

# Arguments for k_fold_cross_validation (train)
train = False
k_fold_path = os.path.join('..', 'Train')  #[path], relative path to Train folder
k = 1 #len(get_class_list(db_path))  #[int], number of folds to be performed
batch_size = 1024  #[int], size of batch in examples (windows)
shuffle_buffer = 3 * batch_size  #[int], size of the buffer used to shuffle the data
epochs = 530 #130  #[int], number of epochs to be performed during training
path_model = os.path.join('..', 'Model', 'model_1.h5')
if not os.path.exists(os.path.split(path_model)[0]): #create Model folder if it does not exist
    os.mkdir(os.path.split(path_model)[0])


optimizer = 'adam'  #[string or tensorflow.keras.optimizers], optimizer to be used
dropout = 0.5  #[float], between 0 and 1. Fraction of the input units to drop
shuffle_mode = False  # [boolean], if True shuffles train and validation datasets as one dataset, else individually


if proc_end_to_end:
    # today = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    #
    # # TODO when multiple channels will be added, the processing will iterate through
    # #  the list of dict_all_filter_settings in params_preproc.py
    # dict_filenames_and_process_variants = aas.create_end_to_end_all_proc_vars_combinations(list_dict_all_filter_settings[0],
    #                                                                                        root_filename="eq_ed",
    #                                                                                        start_index=0,
    #                                                                                        end_index=None,
    #                                                                                        number_of_filters=no_filters)
    # aas.process_signal_all_variants(dict_filenames_and_process_variants) # TODO maybe add run date to the metadata or name of the processed signals
    # # TODO save the .txt elsewhere to avoid messing with the correctness of the test train split

    aas.process_multiple_signals(list_settings_dict=list_dict_all_filter_settings)


    # copyfile("params_preproc.py", os.path.join(preproc_signals_root_folder, f"params-preproc-{today}.txt"))

    # for d in dict_filenames_and_process_variants:
    #     print("file name in dict_filenames_and_process_variants", d, '-----')
    #     print(len(set(dict_filenames_and_process_variants[d].keys())))
    #     print(dict_filenames_and_process_variants[d])
if create_training_features:
    today = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    feats_extractor = FeatureExtractor(feature_list, feature_dict, variance_type, raw_features, keep_feature_dims)

    aas.create_features_diff_for_training(inst_feature_extractor=feats_extractor,
                                          bool_pre_diff=param_pre_diff, bool_process_entire_signal=True)
    # TODO save the .txt elsewhere to avoid messing with the correctness of the test train split
    copyfile("params_features.py", os.path.join(extracted_features_folder, f"params-features-{today}.txt"))

# aas.process_signal_all_variants(signal_in, {test_fname: dict_filenames_and_process_variants[test_fname]})
# training_data = aas.load_labels_metadata_for_training(preproc_signals_root_folder)
# path = r'F:\PCON\Disertatie\AutoMixMaster\datasets\diverse-test\white-noise.wav'
if split:
    split_dataset(extracted_features_folder, split_perc_train) # TODO put the split paths somewhere else

scaler = None
if train:
    if scaler == None:
        scaler = compute_scaler()
    train_cardinality = None
    k_fold_cross_validation(k_fold_path, k, path_model, batch_size, epochs, optimizer, dropout, scaler, shuffle_mode)