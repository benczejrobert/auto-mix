small_no = 1e-320 # delay for unicity of timestamps

# TODO add logarithmic penalty for the frequency parameters, maybe for db gain too (depending on the range of errors)
# TODO add r2 metrics and MSE and others for the evaluation of the model


# TODO check how to do gaussian sampling fo generate data && how to send this into a DAW maybe

# Signal (pre)processor

# sig_root_path = "D:\\PCON\\Disertatie\\AutoMixMaster\\datasets\\diverse-test\\white-noise-mono.wav"
# sig_root_path = "D:\\PCON\\Disertatie\\AutoMixMaster\\datasets\\diverse-test\\resampled_white_noise.wav"
# TODO when multiple channels will be added, sig_root_path will point to a folder with multiple signals
# sig_path = "..\\data\\raw-audio\\resampled_white_noise.wav"
# sig_path = "D:\\PCON\\Disertatie\\AutoMixMaster\\datasets\\diverse-test\\white-noise-reaper-generated.wav"
sig_root_path = "..\\data\\raw-audio\\"

# todo here will be a subfolder structure for each drum channel
preproc_signals_root_folder = "..\\data\\processed-audio-latest"
sample_rate = 22050
# TODO make start index a list so u can choose it for each channel
param_start_index = 0  # left at 65613 for commented generic params Moro

# Usage tips: You need to add numbers at the end of every signal processing type, because
# you can have multiple of the same type such as peak1, peak2, peak3 etc. - always name them with numbers at the end

# Usage tips: include dbgain 0 if you want to ignore a certain type of filter OR remove it from the below dict.

# TODO keep in mind that the order of files is important.
#  If files/channel wavs are not numbered in the sig_root_path,
#  the processing dicts might be applied in a different order.

# TODO maybe try using different ranges
#  for each channel rather than a generic set of params

# gen_hp_freq = list(range(30, 100, 10))
# gen_hp_freq.extend(range(150, 400, 50))
#
# gen_ls_freq = range(100, 301, 50)
# gen_p1_freq = list(range(200, 1001, 100))  # does not cover the kick's peak
# kick_q_interval = [1 / i for i in range(1, 4)]
# gen_gain_interval = list(range(-12, 13, 3)) # checkif any gains lower than -12 in reaper (and if not too much data will be created)
#
# gen_p2_freq = list(range(1000, 5001,500)) # does not cover the kick's peak
# # high shelf
# gen_hs_freq = list(range(5000,10001,1000))
#
#
# # low pass log-ish interval
# gen_lp_freq = list(range(100,451,50))  # generates 1.6 million files until here
# gen_lp_freq.extend(range(500, 1501, 100)) # generates 16 million files until here
# gen_lp_freq.extend(range(1500, 5001, 500))  # 118 million files until here
# gen_lp_freq.extend(range(5000, 10001, 1000))  # 1.18 billion files until here

# gen_lp_freq.extend(range(5000, 15001, 1000))  # if no resampling is considered


# TODO check intervals for the frequencies to match the snare and kick freqs

# TODO also check how to make sure that the channels correpsond to their filter settings within the list.
#  maybe do a dict and use keys as file names or some identifier found in the file name (like regex match) or skip step if no match

general_ls_freq = range(150, 301, 60)
general_q_interval = [0.5, 1]
general_gain_interval = [-9, -5, 0, 5, 9]  # checkif any gains lower than -12 in reaper (and if not too much data will be created)

kick_hp_freq = list(range(30, 101, 15))
kick_hp_freq.extend(range(150, 251, 50))

kick_p1_freq = list(range(50, 351, 50))

kick_p2_freq = list(range(400, 801,100))

### snr-specific params

snr_hp_freq = list(range(100, 341, 60))

snr_p1_freq = list(range(400, 801,100))

snr_p2_freq = list(range(4000, 9300,900))

# elements will correspond to the sorted() files in the raw-audio folder
list_dict_all_filter_settings = [
    {  # kick frequency settings
        "high_pass": {"cutoff": kick_hp_freq, "resonance": [0.5]},
        "low_shelf": {"cutoff": general_ls_freq, "resonance": general_q_interval, "dbgain": general_gain_interval},
        "peak1": {"center": kick_p1_freq, "resonance": general_q_interval, "dbgain": general_gain_interval},
        "peak2": {"center": kick_p2_freq, "resonance": general_q_interval, "dbgain": general_gain_interval},
        "low_pass": {"cutoff": [20000], "resonance": [0.5]},  # maybe ignore for the purpose of the kick in and snr top
        "high_shelf": {"cutoff": [20000], "resonance": [1], "dbgain": [0]}  # maybe ignore for the purpose of the kick in and snr top. TODO decrease the sr threshold
    },  # for kick in and snr top. nu sunt filtre peste 10k in reaper, deci resample ok.  840k files
    {  # snare frequency settings
        "high_pass": {"cutoff": snr_hp_freq, "resonance": [0.5]},
        "low_shelf": {"cutoff": general_ls_freq, "resonance": general_q_interval, "dbgain": general_gain_interval},
        "peak1": {"center": snr_p1_freq, "resonance": general_q_interval, "dbgain": general_gain_interval},
        "peak2": {"center": snr_p2_freq, "resonance": general_q_interval, "dbgain": general_gain_interval},
        "low_pass": {"cutoff": [20000], "resonance": [0.5]},  # maybe ignore for the purpose of the kick in and snr top
        "high_shelf": {"cutoff": [20000], "resonance": [1], "dbgain": [0]}  # maybe ignore for the purpose of the kick in and snr top. TODO decrease the sr threshold
    }  # 450k files
]

dict_normalization_values = { "dbgain_min": -40,
                              "dbgain_max": 40,
                              "freq_min": 20,
                              "freq_max": 20000,
                              "resonance_min": 0,
                              "resonance_max": 10}
dict_params_order = {
    "1_high_pass": ["cutoff", "resonance"],
    "2_low_shelf": ["cutoff", "resonance", "dbgain"],
    "3_peak1": ["center", "resonance", "dbgain"],
    "4_peak2": ["center", "resonance", "dbgain"],
    "5_low_pass": ["cutoff", "resonance"],
    "6_high_shelf": ["cutoff", "resonance", "dbgain"]

}

# Change this to the number of filters you want to use or None
# to use all possible combinations of filters, any number of filters.

