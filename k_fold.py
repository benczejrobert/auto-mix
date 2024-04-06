from utils import *

def k_fold(path, k, fold_nr):
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


    pathology_folders = sorted(os.listdir(path))
    x_train, x_val, y_train, y_val = [], [], [], []
    for pathology in pathology_folders:
        files = sorted(os.listdir(os.path.join(path, pathology)))
        random.shuffle(files)
        if k == 1:
            nr_of_files_to_load = len(files) // 5  #one speaker val hardcoded
            #todo un-hardcode this //5, make it user-specified^
        else:
            nr_of_files_to_load = len(files) // k
        range_min = (fold_nr - 1) * nr_of_files_to_load
        range_max = range_min + nr_of_files_to_load
        eval_files = files[range_min:range_max]
        for file in files:
            npy = np.load(os.path.join(path, pathology, file))
            if file in eval_files:
                x_val.extend(npy[0]) # TODO or append?
                y_val.extend(npy[1])
            else:
                x_train.extend(npy[0])
                y_train.extend(npy[1])
    y_train = np.asanyarray(y_train)
    y_val = np.asanyarray(y_val)
    return (np.asanyarray(x_train), np.asanyarray(x_val),
            np.asanyarray(y_train), np.asanyarray(y_val))
