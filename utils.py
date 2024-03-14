from imports import *

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
    frequency = np.linspace(0, rate, len(magnitude))
    plt.plot(frequency, magnitude)
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

#TODO motivul pentru care se fute bpm-ul e pentru ca eu extrag bpm-ul cu un extractor prost de bpm

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