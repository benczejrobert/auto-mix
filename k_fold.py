from utils import *

def k_fold(path, k, fold_nr, perc_val_single_fold=0.2):
    '''
    Splits the Train set using k-fold principle.

    Input:
        - path [string], relative path to 'Train' folder
        - k [int], number of sub-sets to be generated
        - fold_nr [int], the sub-set to be used as validation data
                        ! must be between 1 and k !
    Output:
        - x_train [2D-array], features used as training data,
                            size: #pathologies * (k - 1)folds * #windows x #features
        - x_val [2D-array], features used as validation data,
                            size: #pathologies * folds * #windows x #features
        - y_train [1D-array], labels for the training data,
                            size: #pathologies * (k - 1)folds * #windows
        - y_val [1D-array], labels used for validation data,
                            size: #pathologies * folds * #windows
    '''
    if fold_nr > k or fold_nr <= 0:
        raise Exception("Incorect value for fold_nr")
    x_train, x_val, y_train, y_val = [], [], [], []  # TODO should only iterate files not dirs.
    for current_filepath, dirs, files in os.walk(path):
        if len(dirs):
            raise Exception(f"{debugger_details()} There should be no directories in the {current_filepath} Train folder")
        if not len(files):
            continue
        crt_sorted_files = sorted(files)
        random.shuffle(crt_sorted_files)
        if k == 1:
            nr_of_files_to_load = round(len(files) * perc_val_single_fold)
        else:
            nr_of_files_to_load = len(crt_sorted_files) // k
        range_min = (fold_nr - 1) * nr_of_files_to_load
        range_max = range_min + nr_of_files_to_load
        eval_files = crt_sorted_files[range_min:range_max]
        for file in crt_sorted_files:
            npy = np.load(os.path.join(current_filepath, file), allow_pickle=True)
            if file in eval_files:
                x_val.append(npy[0])  #TODO append to assoc train datapoints with their labels.
                y_val.append(npy[1])   #but is there a situation where extend would suit?
            else:
                x_train.append(npy[0])
                y_train.append(npy[1])
        # print(debugger_details(), np.asanyarray(x_train).shape, np.asanyarray(y_train).shape)
        # print(debugger_details(), np.asanyarray(x_val).shape, np.asanyarray(y_val).shape)
    y_train = np.asanyarray(y_train)
    y_val = np.asanyarray(y_val)
    return (np.asanyarray(x_train), np.asanyarray(x_val),
            np.asanyarray(y_train), np.asanyarray(y_val))
