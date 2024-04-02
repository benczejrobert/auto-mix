from imports import *


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
        interm = 0.001 * np.float64(interm == 0) + interm
        C = IFFT(np.log(interm), N_fft)[:, 0:window_length]
    except:
        interm = amplitude(FFT(w_signal, N_fft))
        interm = 0.001*np.float64(interm==0) + interm
        C = IFFT(np.log(interm), N_fft)[0:window_length]

    return C.real

def create_model(data = np.array([[[1,2,3],[1,2,3]]]), no_classes = 3, optimizer = 'adam', dropout_rate=0.5, summary=True): #_initial
    inshape = list(data.shape)[1::]
    input = Input(shape=inshape)
    input2 = input
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
    # hdn1 = Dense(512, name='layer1')(input2)
    # act1 = Activation('relu')(hdn1)
    # act1 = BatchNormalization()(act1)
    # dp1 = Dropout(dropout_rate)(act1)
    #
    # hdn2 = Dense(256, name='layer2')(dp1)
    # act2 = Activation('relu')(hdn2)
    # # bn2 = BatchNormalization()(act2)
    # dp2 = Dropout(dropout_rate)(act2)
    #
    # hdn3 = Dense(128, name='layer3')(dp2)
    # act3 = Activation('relu')(hdn3)
    # # bn3 = BatchNormalization()(act3)
    # dp3 = Dropout(dropout_rate)(act3)
    #
    # hdn4 = Dense(64, name='layer4')(dp3)
    # act4 = Activation('relu')(hdn4)
    # # bn4 = BatchNormalization()(act4)
    # dp4 = Dropout(dropout_rate)(act4)
    #
    # hdn5 = Dense(32, name='layer5')(dp4)
    # act5 = Activation('relu')(hdn5)
    # # bn5 = BatchNormalization()(act5)
    # dp5 = Dropout(dropout_rate)(act5)
    hdn6 = Dense(no_classes)(input2)
    output = Activation('softmax')(hdn6)

    model = Model(inputs=input, outputs=output)

    if summary:
        print(model.summary())

    model.compile(optimizer=optimizer, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
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