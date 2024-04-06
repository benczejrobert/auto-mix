from utils import *

def split_dataset(split_path, train_split_perc):
    '''
    Script that splits the dataset in Train and Test features.

    Arguments:
        - path [string], relative path to the 'Features' folder
        - split_perc [int], percentage used to split Features in (split_perc) Train
        			and (1 - split_perc) Test

    '''
    ordinal = ['st','nd','rd']
    # todo may need to be rethinked - Test/Train would be root folder and
    #  split path (or split root if renamed as such) would be in it because
    #  it would be the folder with features for each drum channel/signal

    # TODO in the features folder there will be subfolders for each drum channel.
    #  in the test/train folders there will be the same subfolders with the same files
    test_path = os.path.join(os.sep.join(split_path.split(os.sep)[0:-1]), 'Test')
    train_path = os.path.join(os.sep.join(split_path.split(os.sep)[0:-1]), 'Train')

    # drum_channels = load_channels(split_path) # channels are the subfolders in the split_path i.e. feature_path

    for t_path in [test_path, train_path]:
        if os.path.exists(t_path):
            rmtree(t_path)  # delete Test/Train folder if it exists
        os.mkdir(t_path)  # creates Test/Train folder where the subfolders will be for each of the drum_channels
        # [os.mkdir(os.path.join(t_path, folder)) for folder in drum_channels]  # make the subfolders also
    single_channel = True
    for subdir, dirs, files in os.walk(split_path):
        print(subdir, dirs, files)
        if subdir == split_path and len(files) == 0: # TODO checkif 'and' should be 'or'
            print("in split_dataset subdir = ", subdir)
            single_channel = False
            continue  # skip first iteration (parent folder), get to subfolders
        no_of_test_files = len(files) - int(train_split_perc / 100 * len(files))

        test_file_indices = np.random.choice(np.linspace(0, len(files) - 1, len(files), dtype='int'),
                                             replace=False, size=no_of_test_files)  # make a list of indices

        # TODO check these when multiple channels are present - might need to remove the file name or something.
        if single_channel:
            # dst_test = os.path.join(test_path, subdir.split(os.sep)[-1])
            # dst_train = os.path.join(train_path, subdir.split(os.sep)[-1])
            dst_test = test_path
            dst_train = train_path
        print(debugger_details(), dst_test, dst_train)
        i = 0
        for file in sorted(files):
            print(debugger_details(), os.path.join(subdir, file))
            if i % 10 < 3 and i not in [10, 11, 12]:
                print("split_dataset reached ", i + 1, ordinal[i % 10] + " file of " + subdir)
            else:
                print("split_dataset reached ", i + 1, "th file of " + subdir)
            # if int(file.split('.')[-2]) in test_file_indices:
            if int(i) in test_file_indices:
                if single_channel:
                    # todo differentiation here may not be needed - rather in the creation of the dst_* path
                    copyfile(os.path.join(subdir, file), os.path.join(dst_test, file))
                else:
                    copy(os.path.join(subdir, file), dst_test)
            else:
                if single_channel:
                    # todo differentiation here may not be needed - rather in the creation of the dst_* path
                    copyfile(os.path.join(subdir, file), os.path.join(dst_train, file))
                else:
                    copy(os.path.join(subdir, file), dst_train)
            i += 1