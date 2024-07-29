# import matplotlib.pyplot as plt

## import run parameters from the params files
from params_features import *
from SignalProcessor import *
from FeatureExtractor import *
from split_dataset import *
from channel_feature_scalers import *
from k_fold_cross_validation import *
from predict import *

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

# TODO look up Disentangle Metric for Informativeness (DCI RF
# Expl) [20] as disentanglement metrics. DCIMIG - paper https://arxiv.org/pdf/2211.02247

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
split = False
split_perc_train = 70

# Arguments for k_fold_cross_validation (train)
train = True
train_data_root = os.path.join('..', 'data', 'Train')  #[path], relative path to Train folder
test_data_root = os.path.join('..', 'data', 'Test')  #[path], relative path to Train folder
k = 1 #len(get_class_list(db_path))  #[int], number of folds to be performed
batch_size = 2  #[int], size of batch in examples (diff features)
shuffle_buffer = 3 * batch_size  #[int], size of the buffer used to shuffle the data
epochs = 530 #130  #[int], number of epochs to be performed during training
path_model = os.path.join('..', 'Model', 'model_1.h5')

# Arguments for predict (test)
predict = True

if not os.path.exists(os.path.split(path_model)[0]): #create Model folder if it does not exist
    os.mkdir(os.path.split(path_model)[0])


optimizer = 'adam'  #[string or tensorflow.keras.optimizers], optimizer to be used
dropout = 0.5  #[float], between 0 and 1. Fraction of the input units to drop
shuffle_mode = False  # [boolean], if True shuffles train and validation datasets as one dataset, else individually


if create_training_features:  # TODO make this into a function of (filename, root_to_move_to, bool_for_pipeline_stage)
    today = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    copyfile("params_features.py", os.path.join(extracted_features_folder, f"params-features-{today}.txt"))

if proc_end_to_end:
    # TODO add a check if the processed signals folder already contains the processed signals
    # aas.process_signal_all_variants(dict_filenames_and_process_variants) # TODO maybe add run date to the metadata or name of the processed signals
    aas.process_multiple_signals(list_settings_dict=list_dict_all_filter_settings)

if create_training_features:
    feats_extractor = FeatureExtractor(feature_list, feature_dict, variance_type, raw_features, keep_feature_dims)
    aas.create_features_diff_for_training(inst_feature_extractor=feats_extractor,
                                          bool_pre_diff=param_pre_diff, bool_process_entire_signal=True)

# aas.process_signal_all_variants(signal_in, {test_fname: dict_filenames_and_process_variants[test_fname]})
# training_data = aas.load_labels_metadata_for_training(preproc_signals_root_folder)
# path = r'F:\PCON\Disertatie\AutoMixMaster\datasets\diverse-test\white-noise.wav'
if split:
    split_dataset(extracted_features_folder, split_perc_train) # TODO put the split paths somewhere else
obj_feature_scalers = ChannelFeatureScalers(train_data_root)  # TODO needs to first check if there is data in the folder OR do the test train split before
# TODO scaler takes too long to compute. maybe load files one by one and get their statistical params like min and max or so
# scaler = None
if train:
    # if scaler == None:
    #     scaler = compute_scaler()   # TODO should data be scaled per channel or per everything?
    k_fold_cross_validation(train_data_root, k, path_model, batch_size, epochs, optimizer, dropout, obj_feature_scalers, shuffle_mode)

if predict:
    run_predict(test_data_root, obj_feature_scalers)

