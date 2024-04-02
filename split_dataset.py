from utils import *

def split_dataset(split_path, split_perc, features_cardinality, tfrecord=False, perspeaker = False):
    '''
    Script that splits the dataset in Train and Test features.

    Arguments:
        - path [string], relative path to the 'Features' folder
        - split_perc [int], percentage used to split Features in (split_perc) Train
        			and (1 - split_perc) Test
        - tfrecord: a boolean that indicates to split the .tfrecord file.

        - features_cardinality: the length of the Features TFRecordDataset.

    Outputs (if tfrecord = True):
        - train_cardinality.txt: Text file containing the length of the Train TFRecordDataset.
        - test_cardinality.txt: Text file containing the length of the Test TFRecordDataset.
    '''
    ordinal = ['st','nd','rd']

    test_path = os.path.join(''.join(split_path.split(os.sep)[0:-1]), 'Test')
    train_path = os.path.join(''.join(split_path.split(os.sep)[0:-1]), 'Train')
    for t_path in [test_path, train_path]:
        if os.path.exists(t_path):
            rmtree(t_path)  # delete Test/Train folder if it exists
        os.mkdir(t_path)  # creates Test/Train folder and subfolder for each pathology
        # [os.mkdir(os.path.join(t_path, folder)) for folder in pathologies]

    no_of_test_files = len(files) - int(split_perc / 100 * len(files))