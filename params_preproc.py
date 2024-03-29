small_no = 1e-320 # delay for unicity of timestamps

# Signal (pre)processor

# sig_path = r'D:/PCON/Disertatie/AutoMixMaster/datasets/diverse-test/white-noise-mono.wav'
sig_path = r'D:/PCON/Disertatie/AutoMixMaster/datasets/diverse-test/resampled_white_noise.wav'
# sig_path = r'D:\PCON\Disertatie\AutoMixMaster\datasets\diverse-test\white-noise-reaper-generated.wav'
preproc_signals_root_folder = r"../processed-audio-latest"
sample_rate = 22050
dict_all_filter_settings = {
    "high_pass": {"cutoff": range(200, 201, 1000), "resonance": range(2, 3)},
    "low_shelf": {"cutoff": range(200, 201, 1000), "resonance": range(2, 3), "dbgain": list(range(12, 13, 11))},
    "peak1": {"center": range(1000, 7001, 3000), "resonance": range(2, 3), "dbgain": list(range(-40, 41, 11))},
    "peak2": {"center": range(8000, 8001), "resonance": range(2, 3), "dbgain": [40]},
    "low_pass": {"cutoff": range(10000, 10001, 1000), "resonance": range(2, 3)},
    "high_shelf": {"cutoff": [9000], "resonance": [2], "dbgain": [0]}
}





