import os.path
from imports import *
from params_preproc import *

# from logger import *

def hist_errors(y_pred, y_true, filter_params, model_name):
    """
    Returns a histogram of errors for each parameter or for entire test set
    """
    # TODO see how u can do this metric even better to like say at which real value this error happens (or do some scaling to the real values)
    if len(y_pred) != len(y_true) != 2:
        raise ValueError(f"{debugger_details()} y_pred and y_true should be of the same length and equal to 2")
    y_diff = y_pred - y_true

    param_names = []
    for filter in filter_params:
        for param in dict_params_order[filter]:
            param_names.append(f"{filter}_{param}")
    # get unique values per columns
    # unique_values = [np.unique(y_diff[:,i]) for i in range(y_diff.shape[-1])]
    create_histograms_2d_array(y_diff, param_names, model_name)
    # TODO also add something that specifies the sample size or how many rows were in the test set.

#deprecated
def create_histograms_2d_array_v1(array, param_names, model_name):
    num_cols = array.shape[1]

    for col_index in range(num_cols):
        column_data = array[:, col_index]
        unique_values, value_counts = np.unique(column_data, return_counts=True)

        # Create a range of indices for the unique values
        indices = np.arange(len(unique_values))
        print("col index", col_index)
        # Create a new figure for each histogram
        plt.figure()
        # Create histogram for unique values
        plt.bar(indices, value_counts, edgecolor='black')
        plt.title(f'Hist of model {model_name} \n'
                  f'Params {param_names[col_index]}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.3f')) # still not working, strings are too long
        plt.xticks(indices, unique_values)  # Set x-axis ticks to unique values
        plt.show()

# deprecated
def create_histograms_2d_array_v0(array,param_names):
    num_cols = array.shape[1]

    for col_index in range(num_cols):
        column_data = array[:, col_index]
        unique_values, value_counts = np.unique(column_data, return_counts=True)

        # Sort unique values and value counts based on unique values
        sorted_indices = np.argsort(unique_values)
        unique_values = unique_values[sorted_indices]
        value_counts = value_counts[sorted_indices]

        # Create histogram for unique values
        plt.figure()
        plt.bar(unique_values, value_counts, edgecolor='black')
        plt.title(f'Hist of {param_names[col_index]}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()

def create_histograms_2d_array(array, param_names, model_name, paper = False):
    num_cols = array.shape[1]
    rows_cols_number = int(np.ceil(np.sqrt(num_cols)))
    fig = make_subplots(rows=rows_cols_number,
                        cols=rows_cols_number
                        )
    for col_index in range(num_cols):
        column_data = array[:, col_index]
        unique_values, unique_counts = np.unique(column_data, return_counts=True)

        # Create a mapping from unique values to indices
        index_to_value = dict(enumerate(unique_values))
        # Convert unique values to indices
        indices = list(index_to_value.keys())

        # print("Column{}: ".format(col_index))
        # print(indices)
        # print(unique_counts)
        # print(list(index_to_value.values()))
        cr_row = int(col_index // np.ceil(np.sqrt(num_cols)) + 1)
        cr_col = int(col_index % np.ceil(np.sqrt(num_cols)) + 1)
        fig.add_trace(go.Bar(x=indices, y=unique_counts, name=f'Param no {col_index} \n {param_names[col_index]}'),
                      row=cr_row,
                      col=cr_col
                      )

        # Set the tick labels on the x-axis to be the unique values
        fig.update_xaxes(tickvals=indices,
                         ticktext=list(index_to_value.values()),
                         showticklabels=False, # TODO maybe format this to show only 2-3 decimals or so
                         row=cr_row,
                         col=cr_col
                         )
        if paper:
            fig.add_annotation(
                text=f'Hist of model {model_name} \n Params {param_names[col_index]}',
                xref='paper',
                yref='paper',
                # xref=f'x{col_index + 1}',
                # yref=f'y{col_index + 1}',
                showarrow=False,
                font=dict(size=10),
                xanchor='left',
                yanchor='top',
                # x=(cr_col - 1) / rows_cols_number, # either calculate these based on the normalized coords with xref and yref paper
                # y=cr_row / rows_cols_number,# or use the col_index refs AND calculate relative position based on the values in the histogram
                x= (cr_col - 1) / rows_cols_number,
                y=1 - ((cr_row - 1) / rows_cols_number), # (0,0) of paper is bottom left not top left
                yshift=10
            )
        else:
            fig.add_annotation(
                text=f'Params {param_names[col_index]}',
                xref=f'x{col_index + 1}',
                yref=f'y{col_index + 1}',
                showarrow=False,
                font=dict(size=10),
                xanchor='left',
                yanchor='bottom',
                x=min(indices),  # calculate these based on the indices and unique_counts lists
                y=max(unique_counts) + 0.1 * max(unique_counts),
                # calculate these based on the indices and unique_counts lists
                # yshift=30
            )
    fig.add_annotation(text=f"Hist of model {model_name} \n",
                       xref='paper',
                       yref = 'paper',
                       showarrow=False,
                       font = dict(size=20),
                       xanchor = 'center',
                       yanchor = 'bottom',
                       x = 0.5,
                       y = 1.05
    )
    fig.show(renderer='browser')


def denormalize_params(denorm_me, dn_dict_params_order=dict_params_order,
                       dict_norm_values=dict_normalization_values,
                       norm_type='0,1'):
    """

    :param denorm_me: np.array
    :param dn_dict_params_order:
    :param dict_norm_values:
    :param norm_type: '0,1' or '-1,1' or '-1, 1 max abs'
    :return:
    """
    if not isinstance(denorm_me, np.ndarray):
        raise ValueError("denorm_me should be a numpy array")

    list_denorm_min = []
    list_denorm_max = []
    for filter in dn_dict_params_order:
        for param in dn_dict_params_order[filter]:
            if param in ["cutoff", "center"]:
                list_denorm_min.append(dict_norm_values["freq_min"])
                list_denorm_max.append(dict_norm_values["freq_max"])
            elif param == "resonance":
                list_denorm_min.append(dict_norm_values["resonance_min"])
                list_denorm_max.append(dict_norm_values["resonance_max"])
            elif param == "dbgain":
                list_denorm_min.append(dict_norm_values["dbgain_min"])
                list_denorm_max.append(dict_norm_values["dbgain_max"])

    # print(f"{debugger_details()} list_denorm_min", list_denorm_min)
    # print(f"{debugger_details()} list_denorm_max", list_denorm_max)
    list_denorm_min = np.array(list_denorm_min[0:denorm_me.shape[-1]])
    list_denorm_max = np.array(list_denorm_max[0:denorm_me.shape[-1]])
    # TODO check width of denorm_me and slice list_denorm_min and list_denorm_max accordingly
    # check which axis should be processed from denorm_me

    if norm_type == '0,1':
        denorm_me = denorm_me * (list_denorm_max - list_denorm_min) + list_denorm_min
        # denorm_me = denorm_me * (list_denorm_max - list_denorm_min) + list_denorm_min
    elif norm_type == '-1,1':
        denorm_me = (denorm_me + 1) * (list_denorm_max - list_denorm_min) / 2 + list_denorm_min
    elif norm_type == '-1, 1 max abs':
        denorm_me = denorm_me * np.max(np.abs(list_denorm_max),np.abs(list_denorm_min))
    return denorm_me
def create_test_npy(path, scaler):
    """Creates x_test and y_test (true labels) from .npy files"""
    x_test, y_true = [], []
    for current_filepath, dirs, files in os.walk(path):
        if len(dirs):
            raise Exception(
                f"{debugger_details()} There should be no directories in the {current_filepath} Test folder")
        if not len(files):
            continue
        crt_sorted_files = sorted(files)
        random.shuffle(crt_sorted_files)
        for file in crt_sorted_files:
            npy = np.load(os.path.join(current_filepath, file), allow_pickle=True)
            x_test.append(npy[0])  # TODO append to assoc train datapoints with their labels.
            y_true.append(npy[1])  # but is there a situation where extend would suit?
    x_test = scaler.transform(x_test)
    return (np.asanyarray(x_test), np.asanyarray(y_true))


def get_train_test_paths(features_path):
    """
        Copies the folder structure from the features_path to the Train and Test folders.
    :param features_path:
    :return:
    """
    test_path = os.path.join(os.sep.join(features_path.split(os.sep)[0:-1]), "Test")
    train_path = os.path.join(os.sep.join(features_path.split(os.sep)[0:-1]), "Train")

    for t_path in [test_path, train_path]:
        if os.path.exists(t_path):
            rmtree(t_path)  # delete Test/Train folder if it exists
        os.mkdir(t_path)  # creates Test/Train folder where the subfolders will be for each of the drum_channels

    return train_path, test_path

def get_callbacks(path_model):
    channel_folder = path_model.split(os.sep)[-2]  # get the channel folder from the path_model
    logdir = f"..\\Log\\{channel_folder}\\log_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)
    model_checkpoint = ModelCheckpoint(path_model, monitor="val_loss", verbose=1, save_best_only=True)
    return [model_checkpoint, tensorboard_callback]

def train_model(model, train_dataset, val_dataset, batch_size, epochs, path_model):
    """
    If tfrecord is True:
        train_dataset and val_dataset must be a tensorflow.data.Dataset.Batch
    Else:
        train_dataset and val_dataset must be a tuple of (features, labels)
    """

    model.fit(x=train_dataset[0], y=train_dataset[1], validation_data=(val_dataset[0], val_dataset[1]),
              batch_size=batch_size, epochs=epochs, verbose=2, callbacks=get_callbacks(path_model=path_model))

    # model.fit(x=train_dataset[0].astype('float32'), y=train_dataset[1].astype('float32'),
    #           validation_data=(val_dataset[0].astype('float32'), val_dataset[1].astype('float32')),
    #           batch_size=batch_size, epochs=epochs, verbose=2, callbacks=get_callbacks(path_model=path_model))
    inp = tf.keras.layers.Input(shape=(28,28,1))
    hdn= tf.keras.layers.Conv2D(64, (3,3), activation='relu')(inp)
    hdn = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(hdn)
    hdn = tf.keras.layers.Flatten()(hdn)
    hdn = tf.keras.layers.Dense(256, activation='relu')(hdn)
    hdn = tf.keras.layers.Dense(128, activation='relu')(hdn)
    hdn = tf.keras.layers.Dense(32, activation='relu')(hdn)
    hdn = tf.keras.layers.Dense(10, activation='softmax')(hdn)
    model = tf.keras.models.Model(inputs=inp, outputs=hdn)
    def __init__(self):
        self.min = None
        self.max = None
    def fit(self, x):  # extracts min/max from the data or whatever needed for scaling
        if not isinstance(x,type(np.array([1,2,3]))):
            x = np.array(x)
        self.min = np.min(x)
        self.max = np.max(x)
    def transform(self,x): # scales its input to whatever min/max was extracted via fit()
        if not isinstance(x,type(np.array([1,2,3]))):
            x = np.array(x)
        x = x+np.abs(self.min)
        self.max+=+np.abs(self.min)
        return x/self.max

class MultiDim_MaxAbsScaler(): #if not in use modify name to end with _vertical
    #this scales everything vertically
    def __init__(self):
        self.max_abs_ = None
    def fit(self,x):  # extracts min/max from the data or whatever needed for scaling
        if not isinstance(x,type(np.array([1,2,3]))):
            x = np.array(x)
        y = np.concatenate(x,axis=0)
        self.max_abs_ = np.max(np.abs(y),axis=0)
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

    np.save(os.path.join(s_path,f"{s_scaler_type}_scaler_values.npy"), scaler.__getattribute__(f"{s_scaler_type}"))  # todo also add the list of files left to process AND pop the current file from the list
    np.save(os.path.join(s_path,f"{s_scaler_type}_remaining_filepaths.npy"), list_of_files)  # todo also add the list of files left to process AND pop the current file from the list
    if backup:
        print("--- save_scaler_details() creating backup files ---")
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

def reevaluate_scaler_type(npy,r_scaler_type):
    if len(npy[0].shape) >= 2:  # if 1st element is not a vector or a scalar
        if r_scaler_type == 'minmax':
            r_scaler = MultiDim_MinMaxScaler()
        else:
            r_scaler = MultiDim_MaxAbsScaler()  # if standard scale or max_abs_ [-1,1]
    else:  # TODO is this required? - will the scaler be undefined here at any point?
        if r_scaler_type == 'minmax':
            r_scaler = MinMaxScaler()  # TODO how was this working before?!
        else:
            r_scaler = MaxAbsScaler()  # if standard scale or max_abs_ [-1,1]
            # TODO this reinstantiates the scaler with empty scaler
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
        print("npy[0].shape = ", tr_npy[0].shape)
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
    if csp_max_abs is not None and csp_remaining_list_of_filepaths is not None:
        csp_scaler.csp_max_abs_ = csp_max_abs
        print(
            f" --- compute_csp_scaler(): For channel {os.path.split(csp_data_path)[-1]}, for scaler parameters in ,{f'{csp_scaler_type}_scaler_values.npy'} there are {len(csp_remaining_list_of_filepaths)} files left to parse--- ")
        if len(csp_remaining_list_of_filepaths) == 0:
            return None  # TODO this will not work for multidim csp_scalers and will need an update
    elif csp_remaining_list_of_filepaths is None:
        print(
            f" --- compute_csp_scaler(): For channel {os.path.split(csp_data_path)[-1]}, remaining filepaths are not yet saved --- ")
        test_train_paths = [csp_data_path.replace("Test", "Train"), csp_data_path]
        if "Train" in csp_data_path:
            test_train_paths = [csp_data_path, csp_data_path.replace("Train", "Test")]
        list_of_filepaths = []
        for path in test_train_paths:  # generate files to load
            crt_filepaths = [os.path.join(path, file) for file in sorted(os.listdir(path))]
            list_of_filepaths.extend(crt_filepaths)
            csp_remaining_list_of_filepaths = list_of_filepaths  # todo check how to treat this - should initially load ALL files for both test and train for csp_scaler computation
    return csp_remaining_list_of_filepaths, csp_scaler, csp_max_abs

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
    X = []
    scaler = get_unidim_scaler_type(scaler_type, with_mean)
    remaining_list_of_filepaths = None
    max_abs, remaining_list_of_filepaths = load_scaler_details(
        os.path.join(scaler_params_root, os.path.split(data_path)[-1]), scaler_type)


    csp_result = check_load_scaler_params(remaining_list_of_filepaths, max_abs, scaler, scaler_type, data_path)
    if csp_result is not None:
        remaining_list_of_filepaths, scaler, max_abs = csp_result
    # print(debugger_details(),"csp_result", csp_result)
    # print(debugger_details(),"remaining_list_of_filepaths is not None and len(remaining_list_of_filepaths) > 0",
    #       remaining_list_of_filepaths is not None and len(remaining_list_of_filepaths) > 0)
    # print(debugger_details(),"isinstance(remaining_list_of_filepaths, np.ndarray)",
    #       isinstance(remaining_list_of_filepaths, np.ndarray))
    # \/---TODO maybe add to check_load_scaler_params() from here---\/
    if remaining_list_of_filepaths is not None and len(remaining_list_of_filepaths) > 0:
        if isinstance(remaining_list_of_filepaths, np.ndarray):
            remaining_list_of_filepaths = remaining_list_of_filepaths.tolist()
        # print(f" --- compute_scaler() moving remaining filepaths to list of filepaths")
        # print(f" --- compute_scaler() moving remaining filepaths to list of filepaths")
        list_of_filepaths = remaining_list_of_filepaths # loaded remaining files from the previous run

    try: # TODO checkif scaler is good to return
        npy = tryload_features(list_of_filepaths[0], scaler_type)
        scaler = reevaluate_scaler_type(npy, scaler_type) # TODO maybe this triggers the bug with "ValueError: setting an array element with a sequence."
    except:
        return scaler
    # /\---TODO maybe add to check_load_scaler_params() until here---/\

    # \/---TODO make this multithreaded from here---\/
    for filepath in list_of_filepaths:
        print(f" --- compute_scaler() reached filepath: {filepath}. remaining files: {len(remaining_list_of_filepaths)} --- ")
        npy = tryload_features(filepath, scaler_type)

        scaler.partial_fit([npy[0]]) # error starts here
        remaining_list_of_filepaths.remove(filepath)
        # /\---TODO make this multithreaded until here---/\
        save_scaler_details(scaler, os.path.join(os.path.split(filepath.replace("Train","Test")
                                                              .replace("Test","scaler-params"))[0]),
                                    scaler_type, remaining_list_of_filepaths, len(remaining_list_of_filepaths) % 5 == 0) # create a backup at evdery 5 steps
# scaler.fit(X) # partial_fit() for large datasets

    save_scaler_details(scaler, os.path.join(os.path.split(filepath.replace("Train","Test").replace("Test","scaler-params"))[0]), scaler_type, remaining_list_of_filepaths)
    return scaler


def IFFT(W_signal, l):
    """
    Arguments:

    W_signal: an array containing the FFT of each window, of size #windows x N_fft
    l: length of each window in the output array of windows
    Outputs:

    w_signal: an array of windows from a signal, of size #windows x window_length
    """
    try:
        w_signal = np.fft.ifft(W_signal, W_signal.shape[-1], axis=-1)[:, 0:l]
    except:
        w_signal = np.fft.ifft(W_signal, W_signal.shape[-1], axis=-1)[0:l]
    return w_signal


def utils_cepstrum(w_signal, N_fft):
    """
    Uses FFT, IFFT and log functions to calculate the Cepstrum
    """

    window_length = np.shape(w_signal)[-1]
    try:
        interm = amplitude(FFT(w_signal, N_fft))
        interm = 0.001 * np.float32(interm == 0) + interm
        C = IFFT(np.log(interm), N_fft)[:, 0:window_length]
    except:
        interm = amplitude(FFT(w_signal, N_fft))
        interm = 0.001*np.float32(interm==0) + interm
        C = IFFT(np.log(interm), N_fft)[0:window_length]

    return C.real

def create_model(data = np.array([[[1,2,3],[1,2,3]]]), no_classes = 3, optimizer = 'adam', dropout_rate=0.5, summary=True): #_initial
    inshape = list(data.shape)[1::]
    input1 = Input(shape=inshape)
    input2 = input1
    if len(inshape)>2:
        #dropout = 0.4
        inshape.append(1)
        inshape = tuple(inshape)
        input2 = Conv2D(input_shape=inshape,filters=1,kernel_size=(3,3),padding='same',data_format="channels_last")(input2) #unspecified
        input2 = Activation('relu')(input2) #how many filters and what kernel size?
        input2 = Conv2D(input_shape=inshape,filters=1,kernel_size=(3,3),padding='same',data_format="channels_last")(input2) #unspecified
        input2 = Activation('relu')(input2)
        # nl (macro layer nonlinearity lookup), nf = no filters, (c1,c2)kernel size. (3,5), (5,3) or (3,3) provided best accuracy
        input2 = Conv2D(input_shape=inshape,filters=1,kernel_size=(3,3),padding='same',data_format="channels_last")(input2) #unspecified
        input2 = BatchNormalization()(input2)   # end macro layer 1
        input2 = MaxPooling2D(pool_size=(4, 4), padding='same')(input2)

        input2 = Conv2D(input_shape=inshape,filters=1,kernel_size=(3,3),padding='same',data_format="channels_last")(input2) #unspecified
        input2 = Activation('relu')(input2) #unspecified by Radu Dogaru
        input2 = Conv2D(input_shape=inshape, filters=1, kernel_size=(3, 3), padding='same',data_format="channels_last")(input2)  # unspecified
        input2 = BatchNormalization()(input2) #end macro layer 2
        input2 = MaxPooling2D(pool_size=(4,4), padding='same')(input2)

        input2 = Conv2D(input_shape=inshape, filters=1, kernel_size=(3, 3), padding='same',data_format="channels_last")(input2)  # unspecified
        input2 = BatchNormalization()(input2)   #end macro layer 3
        input2 = MaxPooling2D(pool_size=(4,4), padding='same')(input2)
        input2 = GlobalAveragePooling2D()(input2)
        input2 = Flatten()(input2)  #RDT feature processing might be the key (see prof. Dogaru paper). this is some sort of spectral feature thing
    else:
        print("in create model inshape = ", inshape)
        input2 = Conv1D(input_shape=inshape,filters=1,kernel_size=1,padding='same',data_format="channels_last")(input2)
        input2 = Flatten()(input2)
        # input2 = Dense(units=inshape[0],name='layer0')(input2) # todo try this instead of above 2 lines
    hdn1 = Dense(512, name='layer1')(input2)
    act1 = Activation('relu')(hdn1)
    act1 = BatchNormalization()(act1)
    dp1 = Dropout(dropout_rate)(act1)

    hdn2 = Dense(256, name='layer2')(dp1)
    act2 = Activation('relu')(hdn2)
    # bn2 = BatchNormalization()(act2)
    dp2 = Dropout(dropout_rate)(act2)

    hdn3 = Dense(128, name='layer3')(dp2)
    act3 = Activation('relu')(hdn3)
    # bn3 = BatchNormalization()(act3)
    dp3 = Dropout(dropout_rate)(act3)

    hdn4 = Dense(64, name='layer4')(dp3)
    act4 = Activation('relu')(hdn4)
    # bn4 = BatchNormalization()(act4)
    dp4 = Dropout(dropout_rate)(act4)

    hdn5 = Dense(32, name='layer5')(dp4)
    act5 = Activation('relu')(hdn5)
    # bn5 = BatchNormalization()(act5)
    dp5 = Dropout(dropout_rate)(act5)
    # output = Dense(no_classes)(input2)
    output = Dense(no_classes)(dp5)
    # output = Activation('softmax')(hdn6)

    model = Model(inputs=input1, outputs=output)

    if summary:
        print(model.summary())

    # model.compile(optimizer=optimizer, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=['mean_squared_error', 'r2_score', 'mean_absolute_error']) # MSE or MAE
    return model


def amplitude(W_signal):
    return np.abs(W_signal)

def FFT(w_signal, N_fft):
    """
    Arguments:


        w_signal: an array of windows from a signal, of size #windows x window_length


        N_fft:  #of points to perform FFT

    Outputs:


        W_signal: an array containing the FFT of each window, of size #windows x N_fft

    """

    W_signal = np.fft.fft(w_signal, N_fft, axis=-1)

    return np.array(W_signal)

def debugger_details():
    """
    Returns the file, function and line number of the caller.
    Usage: print(f"{debugger_details()} variables/text you want to print")
    :rtype: object
    :return:
    """
    cf = currentframe()
    return f"""In code_file {os.path.split(cf.f_back.f_code.co_filename)[-1]},function {cf.f_back.f_code.co_name}(), line {cf.f_back.f_lineno}:"""

def mixtgauss(N, p, sigma0, sigma1):
    '''
    WARNING: this function is not normalized

    gives a mixtuare of gaussian noise
    arguments:
    N: data length
    p: probability of peaks
    sigma0: standard deviation of backgrond noise
    sigma1: standard deviation of impulse noise

    output: x: unnormalized output noise

    '''
    q = np.random.randn(N,1)
    u = q<p
    x = (sigma0 * (1 - u) + sigma1 * u) * np.random.randn(N, 1)

    # TODO lookup is the gaussian mixture actually the sum of PDFs/PMFs or is it the sum of the random variables?

    # TODO are synthetic signals are created from the PDFs/PMFs, not from the random variables?

    # if I want to model a signal/(de)compose it based on PDF/PMFs, should I also add a time shift parameter?

    return x

# write a function that displays a signal with matplotlib
def plot_signal(signal, rate, title = 'Signal', xlabel = 'Time (s)', ylabel = 'Amplitude'):
    time = np.arange(0, len(signal)) / rate
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# plot fft of a signal
def plot_fft(signal, rate, title = 'FFT', xlabel = 'Frequency (Hz)', ylabel = 'Amplitude'):
    fft = np.fft.fft(signal)
    magnitude = np.abs(fft)
    N = len(signal)
    n = np.arange(N)
    T = N / rate
    frequency = n / T

    # display fft of np_arr_out_sig
    plt.figure()
    plt.plot(frequency, magnitude)
    plt.xlim(0, rate//2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
def normalization(x):
    x_norm = x / max(np.abs(x))

    return x_norm  # norm [-1,1]

# todo get no_windows based on sig_len and overlap
# find how many windows would fit in a thread on sig_len and a given no_threads
# find no_samples per thread (based on no_windows_per_thread and overlap) and give each thread the start and end

# example -> [1,2,3,4,5,6,7,8,9,10] with 3 threads and 66% overlap window length 3

# 8 windows to 3 threads. 8/3 = 2.66 -> 3 windows per thread.
# overlap = 66% -> 3*0.66 = 2 overlapping samples per window -> 3-2 = 1 non-overlapping sample per window
# 3 windows per thread with 1 non overlapping sample -> one thread has win_len + non_ov_samples * (no_windows_per_thread - 1) samples

# 1,2,3 # thread 1
# 2,3,4 # thread 1
# 3,4,5 # thread 1

# 4,5,6 # thread 2
# 5,6,7 # thread 2
# 6,7,8 # thread 2

# 7,8,9 # thread 3
# 8,9,10 # thread 3

def sigwin(x, l, w_type = 'rect', overlap = 0):
    """
    w_type[string] can be:  -rect
                            -boxcar
                            -triang
                            -blackman
                            -hamming
                            -hann
                            -bartlett
                            -flattop
                            -parzen
                            -bohman
                            -blackmanharris
                            -nuttall
                            -barthann

    overlap [percentage]
    l[sample number]
    x[list or np.array]
    """
    overlap=overlap/100
    if type(x)==np.ndarray:
        x=x.tolist()
    w = []
    delay = int((1- overlap)*l)

    if( w_type !='rect'):
        win = windows.get_window(w_type,l).tolist()

    for i in range(0, len(x), delay):
        if i+l<=len(x):
            if (w_type == 'rect'):
                w.append(x[i:i+l])
            else:
                w.append(np.multiply(win,x[i:i+l]))

    return np.array(w)

def to_db(absolute,ref=1,power=False):
    return (10*(not power) + 10)*np.log10(np.abs(absolute)/ref)

def to_absolute(db,ref=1,power=False):
    return ref * 10**(db / (10 * (not power) + 10))
#TODO onset detection
# TODO bpm detection
# TODO compare onsets to what should happen if bpm is correct

# TODO trim both
# TODO add funcitonality to count the measure stuff's played in. it starts with 2, if shit's divisible with 2 it saves it and increments. if shit's divisible with 3, it increments.
#  prime factor decomposition I guess (sort of) because you might just have 7 bars of 4. so I suppose you could stop at 7 and if the thing is divisible with both 7&4 then idk what to select.
#  maybe select shit also based on the accented beat, even tho it might not be a correct estimation especially if there's someone without too much control over the intensity of the hits

#TODO pack all this in a nice thing

#TODO what if I do it real-time?

#TODO motivul pentru care se strica bpm-ul e pentru ca eu extrag bpm-ul cu un extractor prost de bpm

def tempo(signal, rate):
    onset_env = librosa.onset.onset_strength(signal, sr=rate)
    return librosa.beat.tempo(onset_envelope=onset_env, sr=rate)

def generate_tempo(bpm, sr):
    beat_interval = 60 / bpm  # Calculate the interval between beats in seconds
    samples_per_beat = int(sr * beat_interval)  # Calculate the number of samples per beat
    total_samples = int(sr)  # Total number of samples for a 1-second signal

    # Generate an array of zeros
    signal = np.zeros(total_samples)

    # Set the first sample to 1
    signal[0] = 1

    # Set the remaining ones at the appropriate intervals
    for i in range(1, total_samples):
        if i % samples_per_beat == 0:
            signal[i] = 1

    return signal

def generate_combinations(input_dict):
    keys = input_dict.keys()
    values = input_dict.values()

    combinations = list(product(*values))

    result = []
    for combo in combinations:
        output_dict = dict(zip(keys, combo))
        result.append(output_dict)

    return result


def squared_median_abs_dev(x):
    if len(x.shape) == 1:
        return scipy.stats.median_absolute_deviation(x)**2
    elif len(x.shape) == 2:
        return np.mean(scipy.stats.median_absolute_deviation(x, axis=1)**2)
    else:
        raise TypeError("Input must be a vector or a matrix")