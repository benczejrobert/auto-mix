# import os.path
from imports import *
from params_preproc import *

# from logger import *
def join(a,b):
    """
        get common values from 2 lists
    """
    return list(set(a) & set(b))

def get_iterable_splits(in_iterable, no_splits):
    list_dict_splits = []
    len_subdict = len(in_iterable) // no_splits
    for i in range(no_splits):
        if i < no_splits - 1:
            list_dict_splits.append(slice(i*len_subdict,(i+1)*len_subdict))
        else:
            list_dict_splits.append(slice(i*len_subdict,None))
    return list_dict_splits

def list_errors(y_pred, y_true, filter_params, model_name, showhist = True):
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
    print("----- Results for model name =",model_name)
    print("\tMSE for parameters is: ",(np.square(np.array(y_pred) - np.array(y_true))).mean(axis=0).tolist())
    print("\tMAE for parameters is: ",(np.abs(np.array(y_pred) - np.array(y_true))).mean(axis=0).tolist())
    print("\t1)Mean Relative absolute error for parameters is: ",(((np.array(y_pred) - np.array(y_true) + 0.00001)/(y_true+0.00001)).mean(axis=0)).tolist())
    print("\t2)Mean Relative absolute error for parameters is: ",(((np.array(y_pred) - np.array(y_true) + 0.00001)/(y_true+0.00001).mean(axis=0)).mean(axis=0)).tolist())
    print("\t3)Mean Relative absolute error for parameters is: ",(((np.array(y_pred) - np.array(y_true)).mean(axis=0)/(y_true+0.00001).mean(axis=0))).tolist())
    print("\t4)Mean Relative absolute error for parameters is: ",(((np.array(y_pred) - np.array(y_true) + 0.00001)/(y_true+0.00001)).mean(axis=0)).tolist())

    if showhist:
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
    # TODO make this function show errors in larger bins rather than unique errors
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

def denorm_param_01(x,lower_bound,upper_bound):
    return x * (upper_bound - lower_bound) + lower_bound

def log_denorm_param_01(x,lower_bound,upper_bound):
    return 10 ** (x * (np.log10(upper_bound) - np.log10(lower_bound)) + np.log10(lower_bound))


def denormalize_params(denorm_me, dn_dict_params_order=dict_params_order,
                       dict_norm_values=dict_normalization_values,
                       norm_type='0,1'):
    # raise Exception("repair me to accept denormalization of logarithmic params as well - only frequency ones")
    # freq params denorm like 10 ** (x * (np.log10(upper_bound) - np.log10(lower_bound)) + np.log10(lower_bound))
    # formula not good atm: denorm_me = base ** (denorm_me * (list_denorm_max - list_denorm_min) + list_denorm_min)
    """
    :param denorm_me: np.array - 2D shape - list of lists that contain the outputs of the model
    :param dn_dict_params_order:
    :param dict_norm_values:
    :param norm_type: '0,1' or '-1,1' or '-1, 1 max abs'
    :return:
    """
    if not isinstance(denorm_me, np.ndarray):
        raise ValueError("denorm_me should be a numpy array")

    list_denorm_min = []
    list_denorm_max = []
    functions = []
    # create functions and min max vectors
    for filter in dn_dict_params_order:
        for param in dn_dict_params_order[filter]:
            if param in ["cutoff", "center"]:
                list_denorm_min.append(dict_norm_values["freq_min"])
                list_denorm_max.append(dict_norm_values["freq_max"])
                functions.append(log_denorm_param_01)
            elif param == "resonance":
                list_denorm_min.append(dict_norm_values["resonance_min"])
                list_denorm_max.append(dict_norm_values["resonance_max"])
                functions.append(denorm_param_01)
            elif param == "dbgain":
                list_denorm_min.append(dict_norm_values["dbgain_min"])
                list_denorm_max.append(dict_norm_values["dbgain_max"])
                functions.append(denorm_param_01)

    # print(f"{debugger_details()} list_denorm_min", list_denorm_min)
    # print(f"{debugger_details()} list_denorm_max", list_denorm_max)
    list_denorm_min = np.array(list_denorm_min[0:denorm_me.shape[-1]])
    list_denorm_max = np.array(list_denorm_max[0:denorm_me.shape[-1]])
    functions = np.array(functions[0:denorm_me.shape[-1]])
    # check which axis should be processed from denorm_me

    if norm_type == '0,1':
        # denorm_me = base ** (denorm_me * (list_denorm_max - list_denorm_min) + list_denorm_min)
        denorm_me_out = np.array([
                            [functions[i](crt_denorm_me[i],list_denorm_max[i],list_denorm_min[i])
                                   for i in range(len(functions))]
                                        for crt_denorm_me in denorm_me])
        # denorm_me = denorm_me * (list_denorm_max - list_denorm_min) + list_denorm_min
    elif norm_type == '-1,1':
        denorm_me_out = (denorm_me + 1) * (list_denorm_max - list_denorm_min) / 2 + list_denorm_min
    elif norm_type == '-1, 1 max abs':
        denorm_me_out = denorm_me * np.max(np.abs(list_denorm_max),np.abs(list_denorm_min))
    return denorm_me_out
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
    # model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=['mean_squared_error', 'R2Sscore', 'mean_absolute_error']) # MSE or MAE
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=['mean_squared_error', 'mean_absolute_error']) # MSE or MAE
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
