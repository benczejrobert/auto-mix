from utils import *

class MultiDim_MaxAbsScaler(): #if not in use modify name to end with _vertical
    """
        This class scales every matrix vertically
        The fit function was designed to operate on lists of matrices,
        where the entire data is a list of matrices
        as opposed to its Sklearn fit() counterpart that would either accept:
        a single matrix - i.e. list of lists
        OR a concatenated matrix.



    """
    def __init__(self):
        self.max_abs_ = None
        self.mabsscaler = MaxAbsScaler()
    def fit(self,x):  # extracts min/max from the data or whatever needed for scaling
        if not isinstance(x,type(np.array([1,2,3]))):
            x = np.array(x)
        y = np.concatenate(x,axis=0)
        self.max_abs_ = np.max(np.abs(y),axis=0)
    def partial_fit(self,x):
        if not isinstance(x,type(np.array([1,2,3]))):
            x = np.array(x)
        if len(x.shape) == 2:
            self.mabsscaler.partial_fit(x)
        elif len(x.shape) == 3 and x.shape[0] == 1:
            self.mabsscaler.partial_fit(x[0])
        else:
            raise Exception("Please input a matrix for partial"
                            " fit or a list of a single matrix (2-D array)."
                            f"data shape = {x.shape}")
        self.max_abs_ = self.mabsscaler.max_abs_
    def transform(self,x): # scales its input to whatever min/max was extracted via fit()
        if not isinstance(x,type(np.array([1,2,3]))):
            x = np.array(x)
        return x/self.max_abs_

class MultiDim_MaxAbsScaler_orig():  #if not in use modify name to end with _original
    def __init__(self):
        self.max_abs_ = None
    def fit(self,x):  # extracts min/max from the data or whatever needed for scaling
        if not isinstance(x,type(np.array([1,2,3]))):
            x = np.array(x)
        self.max_abs_ = np.max(np.abs(x))
    def transform(self,x): # scales its input to whatever min/max was extracted via fit()
        if not isinstance(x,type(np.array([1,2,3]))):
            x = np.array(x)
        return x/self.max_abs_

def save_scaler_details(scaler, s_path, s_scaler_type, list_of_files,backup = False):
    """
    Saves the values of the scaler in a .npy file
    :param scaler: the scaler object
    :param s_path: path where the file will be saved
    :param s_scaler_type: the type of the scaler
    :param status: the status of the parameter computation (e.g. True for the fit data)
    """
    if not os.path.exists(s_path):
        os.makedirs(s_path)

    # TODO make this write at every N iterations AND make the parameters write ONLY at a new max value (if not already like this)
    np.save(os.path.join(s_path,f"{s_scaler_type}_scaler_values.npy"), scaler.__getattribute__(f"{s_scaler_type}"))  # todo also add the list of files left to process AND pop the current file from the list
    np.save(os.path.join(s_path,f"{s_scaler_type}_remaining_filepaths.npy"), list_of_files)  # todo also add the list of files left to process AND pop the current file from the list
    if backup:
        print("--- save_scaler_details() creating backup files ---")
        time.sleep(small_no)
        np.save(os.path.join(s_path,f"bkp_{s_scaler_type}_scaler_values.npy"), scaler.__getattribute__(f"{s_scaler_type}"))  # todo also add the list of files left to process AND pop the current file from the list
        np.save(os.path.join(s_path,f"bkp_{s_scaler_type}_remaining_filepaths.npy"), list_of_files)  # todo also add the list of files left to process AND pop the current file from the list


def load_scaler_values(l_path, s_scaler_type):
    """
    Loads the values of the scaler from a .npy file
    """
    try:
        if not os.path.exists(l_path):
            # create path
            os.makedirs(l_path)
            return None
        elif not os.path.exists(os.path.join(l_path, f"{s_scaler_type}_scaler_values.npy")):
            return None
        else:
            return np.load(os.path.join(l_path, f"{s_scaler_type}_scaler_values.npy"), allow_pickle=True)
    except Exception as e:
        return np.load(os.path.join(l_path, f"bkp_{s_scaler_type}_scaler_values.npy"), allow_pickle=True)
        # if "interpret as a pickle" in str(e):
        #     print(f"load_scaler_values(): The file {os.path.split(l_path)[-1]} is most likely corrupt due to an interruption of the overwriting process")
        # raise e

def load_remaining_scaler_filepaths(l_path, s_scaler_type):
    """
        Loads the remaining filepaths to compute scaler values from a .npy file
        :param l_path:
        :param s_scaler_type:
        :return:
    """
    try:
        if not os.path.exists(l_path):
            # create path
            os.makedirs(l_path)
            return None
        elif not os.path.exists(os.path.join(l_path, f"{s_scaler_type}_remaining_filepaths.npy")):
            return None
        else:
            print(f"{debugger_details()}: return" , f"{s_scaler_type}_remaining_filepaths.npy")
            return np.load(os.path.join(l_path, f"{s_scaler_type}_remaining_filepaths.npy"), allow_pickle=True)  # array of paths no-no WHY

    except Exception as e:
        return np.load(os.path.join(l_path, f"bkp_{s_scaler_type}_remaining_filepaths.npy"),
                allow_pickle=True)  # array of paths no-no WHY
        # if "interpret as a pickle" in str(e):
        #     print(f"load_remaining_scaler_filepaths(): The file {os.path.split(l_path)[-1]} is most likely corrupt due to an interruption of the overwriting process")
        # raise e

def load_scaler_details(ls_path, ls_scaler_type):
    return load_scaler_values(ls_path, ls_scaler_type), load_remaining_scaler_filepaths(ls_path, ls_scaler_type)

def get_unidim_scaler_type(scaler_type, with_mean=True):
    """
        Returns the scaler type based on the input
    :param scaler_type: [string], can be 'standard', 'minmax', 'max_abs_'
    :return:
    """

    if scaler_type not in ['standard', 'minmax', 'max_abs_']:
        raise Exception(f"{debugger_details()} Please select scaler_type from: 'standard', 'minmax', 'max_abs_'. scaler_type is {scaler_type}")
    if scaler_type == 'standard':
        scaler = StandardScaler(
            with_mean=with_mean)  # -> not just a max_abs_ +/-3, also modifies the distribution to Gauss
    if scaler_type == 'minmax': # todo unimplemented/tested for multidim
        scaler = MinMaxScaler()  # -> does indeed [0,1]. BUT requires individual feature vectors of max. 1D shape
    if scaler_type == 'max_abs_':
        scaler = MaxAbsScaler()  # -> does indeed [-1,1]
    return scaler

def evaluate_scaler_type(r_filepath,r_scaler_type, r_with_mean=True):
    npy = tryload_features(r_filepath, r_scaler_type)
    if len(npy[0].shape) >= 2:  # if 1st element is not a vector or a scalar
        if r_scaler_type == 'minmax':
            r_scaler = MultiDim_MinMaxScaler()
        else:
            r_scaler = MultiDim_MaxAbsScaler()  # if standard scale or max_abs_ [-1,1]
    else:  # TODO is this required? - will the scaler be undefined here at any point?
        r_scaler = get_unidim_scaler_type(r_scaler_type, r_with_mean)
    return r_scaler
def tryload_features(t_filepath, t_scaler_type='max_abs_'):
    """
        This function tries to load the features from a file. If it fails, it will ask the user to replace the file.
    :param t_scaler_type:
    :param t_filepath:
    :return:
    """
    try:
        tr_npy = np.load(t_filepath,
                         allow_pickle=True)  # this contains the features and the labels. get only features
        # print("npy[0].shape = ", tr_npy[0].shape)
        # tr_npy = tr_npy[0]
        # if True: # TODO implement if the scalers have not been already loaded (i.e. 1st run) OR if the scaling param is a single numbern. otherwise this will only reset the value of the scaling parameter (e.g. max_abs_)
        #     r_scaler = reevaluate_scaler_type(tr_npy, t_scaler_type)
    except Exception as e:  # 1st element is a scalar, scalars have no len()
        print("Exception for file at path = ", t_filepath)
        print(f"{debugger_details()} Exception: {e}")
        input("Replace the file and press enter to continue")
        tr_npy = tryload_features(t_filepath)
    finally:
        return tr_npy
    # return (t_scaler, npy)

def check_load_scaler_params(csp_remaining_list_of_filepaths,
                        csp_max_abs, csp_scaler, csp_scaler_type, csp_data_path):
    csp_list_of_filepaths = []
    if csp_max_abs is not None and csp_remaining_list_of_filepaths is not None:
        csp_scaler.csp_max_abs_ = csp_max_abs
        print(
            f" --- compute_csp_scaler(): For channel {os.path.split(csp_data_path)[-1]}, for scaler parameters in ,{f'{csp_scaler_type}_scaler_values.npy'} there are {len(csp_remaining_list_of_filepaths)} files left to parse--- ")
        if len(csp_remaining_list_of_filepaths) == 0:
            return csp_remaining_list_of_filepaths, csp_list_of_filepaths, csp_scaler, csp_max_abs  # TODO this will not work for multidim csp_scalers and will need an update
    elif csp_remaining_list_of_filepaths is None:
        print(
            f" --- compute_csp_scaler(): For channel {os.path.split(csp_data_path)[-1]}, remaining filepaths are not yet saved --- ")
        test_train_paths = [csp_data_path.replace("Test", "Train"), csp_data_path]
        if "Train" in csp_data_path:
            test_train_paths = [csp_data_path, csp_data_path.replace("Train", "Test")]
        for path in test_train_paths:  # generate files to load
            crt_filepaths = [os.path.join(path, file) for file in sorted(os.listdir(path))]
            csp_list_of_filepaths.extend(crt_filepaths)
            csp_remaining_list_of_filepaths = csp_list_of_filepaths  # todo check how to treat this - should initially load ALL files for both test and train for csp_scaler computation
    csp_remaining_list_of_filepaths, csp_list_of_filepaths = reevaluate_filepath_lists(csp_remaining_list_of_filepaths, csp_list_of_filepaths)
    return csp_remaining_list_of_filepaths, csp_list_of_filepaths, csp_scaler, csp_max_abs
def reevaluate_filepath_lists(rev_remaining_list_of_filepaths, rev_list_of_filepaths):
    if rev_remaining_list_of_filepaths is not None and len(rev_remaining_list_of_filepaths) > 0:
        if isinstance(rev_remaining_list_of_filepaths, np.ndarray):
            rev_remaining_list_of_filepaths = rev_remaining_list_of_filepaths.tolist()
        rev_list_of_filepaths = rev_remaining_list_of_filepaths.copy() # loaded remaining files from the previous run
    return rev_remaining_list_of_filepaths, rev_list_of_filepaths
def compute_scaler(data_path, with_mean=True, scaler_type='max_abs_'):
    """
    Computes the scaler on the entire database for a current channel.

    Arguments:
        - data_path [string], relative path to the data folder, e.g. '..\\data\\Train\\Kick-In'
        - tfrecord - NOT USED - [boolean], if true, reads data from TFRecord files, else from .npy files
        - scaler_type [string] can be 'standard', 'minmax', 'max_abs_'
    Output:
        - scaler [a fitted sklearn.preprocessing scaler]
        Can be Standard -> [mean - 3*sigma, mean + 3*sigma] , MinMax -> default [0,1]  or MaxAbs -> [-1,1]

    @BR20240620: partial_fit was used instead of fit because
    """
    scaler_params_root = os.path.join("..","data","scaler-params")
    scaler = evaluate_scaler_type(os.path.join(data_path, os.listdir(data_path)[0]), scaler_type,
                                  with_mean)  # TODO maybe this triggers the bug with "ValueError: setting an array element with a sequence."
    # reevaluate scaler may reset scaler parameters and yield inconsistent results (i.e. inconsistent max abs) - solved. renamed to evaluate_scaler_type

    max_abs, remaining_list_of_filepaths = load_scaler_details(
        os.path.join(scaler_params_root, os.path.split(data_path)[-1]), scaler_type)



    remaining_list_of_filepaths, list_of_filepaths, scaler, max_abs = (
        check_load_scaler_params(remaining_list_of_filepaths, max_abs, scaler, scaler_type, data_path))

    for filepath in list_of_filepaths:
        # TODO instead of remaining files list with remove element, rather
        #  save last file index AND access the files by file index so
        #  u just get the root file name + index
        print(f" --- compute_scaler() reached filepath: {filepath}. remaining files: {len(remaining_list_of_filepaths)} --- ")
        npy = tryload_features(filepath, scaler_type) # can be parallelized
        # print(npy[0].shape)
        scaler.partial_fit([npy[0]]) # can be parallelized? - yes, get a max_abs_ then do a partial fit on the resulting list of max_abs_ values. make scaler shared between threads
        remaining_list_of_filepaths.remove(filepath) # requires a lock and make variable shared between threads
        # /\---TODO make this multithreaded until here---/\
        # TODO maybe add a status flag if partial fit has changed the scaler parameters
        save_scaler_details(scaler,
                            os.path.join(os.path.split(filepath.replace("Train","Test")
                                                       .replace("Test","scaler-params"))[0]), # create a backup at evdery 5 steps.
                            scaler_type, remaining_list_of_filepaths,
                            backup=len(remaining_list_of_filepaths) % 5 == 0) # requires a lock
    try: # if list_of_filepaths is empty, filepath is not initialized, so this step should be skipped
        save_scaler_details(scaler, os.path.join(os.path.split(filepath.replace("Train","Test").replace("Test","scaler-params"))[0]), scaler_type, remaining_list_of_filepaths)
    except:
        # max_abs, remaining_list_of_filepaths = load_scaler_details(
        #     os.path.join(scaler_params_root, os.path.split(data_path)[-1]), scaler_type)
        scaler.partial_fit([max_abs])
    return scaler


class ChannelFeatureScalers:
    def __init__(self, trian_data_root, list_scaler_types = None, list_with_mean = None):
        """

        :param trian_data_root:  - relative path to Train folder
        :param list_scaler_types: - list of scaler types to be used for each channel (see compute_scaler() function in utils.py)
        :param list_with_mean:  - list of boolean values indicating whether to use mean for each channel
        """
        self.scalers = {}
        self.train_data_root = trian_data_root
        self.list_scaler_types = list_scaler_types
        self.list_with_mean = list_with_mean
        if list_scaler_types is not None:
            if not len(list_scaler_types) == len(os.listdir(self.train_data_root)):
                raise Exception("The length of list_scaler_types and trian_data_root should be equal")
        else:
            self.list_scaler_types = ['max_abs_'] * len(os.listdir(self.train_data_root))
        if list_with_mean is not None:
            if not len(list_with_mean) == len(os.listdir(self.train_data_root)):
                raise Exception("The length of list_with_mean and trian_data_root should be equal")
        else:
            self.list_with_mean = [True] * len(os.listdir(self.train_data_root))
        # todo skip this if the crt channel is already scaled or sth
        for current_filepath, dirs, files in sorted(os.walk(self.train_data_root)):
            i = 0
            if not len(files):
                print("CONTINUE!!!")
                continue
            channel = os.path.split(current_filepath)[-1]
            print(f"Computing scaler for channel {channel} and path {current_filepath}")
            self.scalers[channel] = compute_scaler(current_filepath, with_mean=self.list_with_mean[i],
                                                   scaler_type=self.list_scaler_types[i])
            i += 1
