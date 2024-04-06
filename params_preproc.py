small_no = 1e-320 # delay for unicity of timestamps

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
# TODO when multiple channels will be added, this dict can be added to a list of dicts, one for each channel

# Usage tips: You need to add numbers at the end of every signal processing type, because
# you can have multiple of the same type such as peak1, peak2, peak3 etc. - always name them with numbers at the end

# Usage tips: include dbgain 0 if you want to ignore a certain type of filter OR remove it from the below dict.
# TODO keep in mind that the order of files is important.
#  If files/channel wavs are not numbered in the sig_root_path,
#  the processing dicts might be applied in a different order.
list_dict_all_filter_settings = [{
    "high_pass": {"cutoff": range(200, 201, 1000), "resonance": range(2, 3)},
    "low_shelf": {"cutoff": range(200, 201, 1000), "resonance": range(2, 3), "dbgain": list(range(12, 13, 11))},
    "peak1": {"center": range(1000, 7001, 3000), "resonance": range(2, 3), "dbgain": list(range(-40, 41, 11))},
    "peak2": {"center": range(8000, 8001), "resonance": range(2, 3), "dbgain": [40]},
    "low_pass": {"cutoff": range(10000, 10001, 1000), "resonance": range(2, 3)},
    "high_shelf": {"cutoff": [9000], "resonance": [2], "dbgain": [0]}
}] * 2

dict_normalization_values = { "dbgain_min": -40,
                              "dbgain_max": 40,
                              "freq_min": 20,
                              "freq_max": 20000,
                              "resonance_min": 0,
                              "resonance_max": 10}

# Change this to the number of filters you want to use or None
# to use all possible combinations of filters, any number of filters.

