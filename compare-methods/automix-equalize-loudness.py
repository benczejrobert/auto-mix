import os.path

from utils import *
import numpy as np
import pymixconsole as pymc
import soundfile as sf
import pyloudnorm as pyln
def to_db(amplitude,ref=1,power=False):
    return (10*(not power) + 10)*np.log10(np.abs(amplitude)/ref)

def to_amplitude(db,ref=1,power=False):
    return ref * 10**(db / (10 * (not power) + 10))

#TODO lookup which paper it comes from - automix params with conv1d, batch norm, FiLM, PReLU
root = os.path.join(*["..","AutoMixMaster","datasets","ENST-drums-audio","ENST-drums-public","drummer_1","audio"])
file_name = "036_phrase_disco_simple_slow_sticks.wav"
mix_path = os.path.join(root,"dry_mix",file_name)
_, rate = sf.read(mix_path) #TODO maybe use this as ref? - nah really because it's processed otherwise than simply summing the audios.
dir_list = os.listdir(root) # TODO reupload  np.load(os.path.join("../data/Train/Kick-In", "diff_features_and_params_Kick-In_eq_ed_510741.npy"),allow_pickle=True)
for f in ["wet_mix","accompaniment",".DS_Store","dry_mix"]:
    dir_list.remove(f)
audios = []
rates = [] # wrong to equalize all the mix.
# un track care are un tom pe care il lovesti o sg data, are o problema
# bleed va fi dat tare pe sectiunile incete - this should not happen!!!
for f in dir_list:
    to_load = os.path.join(root,f,file_name)
    data, rate = sf.read(to_load) # load audio (with shape (samples, channels))
    audios.append(data)
    rates.append(rate)
lens = []
def to_samples(ms,rate):
    return int(rate*ms/1000)


def diff(sig_to_proc,ref_partial_loudness_db,_meter,verbose=False, sig_ref_of_full_loudness = None):
    crt_partial_loudness_db = _meter.integrated_loudness(sig_to_proc)
    # 2) diff crt loudness and ref per track
    # 3) diff to ampl
    if ref_partial_loudness_db == -np.inf:
        ampl_loudness = 0
        diff_loudness_db = np.inf
    elif crt_partial_loudness_db == -np.inf:
        ampl_loudness = to_amplitude(ref_partial_loudness_db)
        diff_loudness_db = -ref_partial_loudness_db
    else:
        diff_loudness_db = crt_partial_loudness_db - ref_partial_loudness_db
    diff_loudness_ampl = to_amplitude(-diff_loudness_db)
    if verbose:
        if ref_partial_loudness_db != -np.inf:
            print("\t---crt, ref, full ref", crt_partial_loudness_db, ref_partial_loudness_db,
                  meter.integrated_loudness(sig_ref_of_full_loudness)) # sig_ref (e gen sig ref full) mix_non_eq[i:i + l])
        if ref_partial_loudness_db != -np.inf:
            print("\tref and corrected loudness", ref_partial_loudness_db, _meter.integrated_loudness(diff_loudness_ampl * sig_to_proc))
            # unele parti de semnal partial pot fi cu 0. e normal pe ele sa ai -inf
    return diff_loudness_ampl * sig_to_proc

milliseconds = 500
meter = pyln.Meter(rate=rate,block_size=milliseconds/1000) # create BS.1770 meter
l = to_samples(milliseconds,rate)
delay = int(l)
mix_eq = []
audios_to_pad = np.array(audios)
audios = np.zeros((audios_to_pad.shape[0],l*((audios_to_pad.shape[1]+l)//l)))
audios[:audios_to_pad.shape[0],:audios_to_pad.shape[1]] = audios_to_pad
mix_non_eq = np.sum(audios,axis=0)
# normez fiec track individual
# apoi normez si suma

# TODO lookup articol partial loudness

# TODO eventual baga un ratio in compresor si fa ceva procent din diferenta, de ex. 50% diff
# TODO write signal and listen to it
#  eventual bag o ratie ca sa vad cate chestii suna prea tare/prea incet


no_tracks = audios.shape[0]
# TODO add zero-padding to audios and mix to make them all multiples of 500 ms
for i in range(0, audios.shape[1], delay):
    # normez suma, impart la 7 si apoi transf in db
    # norm_maxAbs(mix_non_eq), div 7, si diferenta dintre crt_track si raport adaug sau scad la fiecare track
    # TODO rename normat/normalized in loc de equalized

    # 1) divide ref_loudness_ampl by n_tracks - that is reference loudness per track
    crt_ref_loudness_db = meter.integrated_loudness(mix_non_eq[i:i + l]/7) # in scala logaritmica nu ar trebui sa impart la 7. #TODO de vazut cum faci medie pe logaritm.
    # sau putem lua un standard (de ex. cinema, -23 sau -24 LUFS)

    audio_sum = np.zeros(audios[0,i:i+l].shape)
    # TODO this will need each signal to be processed individually based on the loudness of the crt mix window
    k = 0 ###
    for a in audios:
        k += 1
        print("crt_index = ",k)
        crt_processed_sig = diff(a[i:i + l],crt_ref_loudness_db,meter,True,mix_non_eq[i:i + l])
        audio_sum = audio_sum + crt_processed_sig #TODO something seems wrong here
    #TODO then make a normalized copy and also a non normalized copy
    if i + l <= audios.shape[1]:
        mix_eq.extend(audio_sum)
    if crt_ref_loudness_db != -np.inf:
        print("crt sum vs ref",meter.integrated_loudness(audio_sum),crt_ref_loudness_db)
        #TODO mai fac un diff pe aici ca sa egalizez si mai inmultesc in amplitudine

        # TODO fa verificarile matematice daca media pe aamplitudine se traduce in medie pe logaritm
        #  ca sa pot face medie, fie normez in scara liniara (google said)
    # if ref_loudness_db != -np.inf:
    #     break

sf.write("test_mix_non_equal_loudness.wav",np.array(mix_non_eq),rate)
sf.write("test_mix_equal_loudness.wav",np.array(mix_eq),rate)

## TODO google this:

"""
Convert Decibel Values to Linear Scale:
If the decibel values are based on power ratios (e.g., signal power), the conversion is:Linear Value=10dB Value10Linear Value=1010dB Value
If the decibel values are based on amplitude ratios (e.g., voltage, pressure), the conversion is:Linear Value=10dB Value20Linear Value=1020dB Value
Calculate the Mean of the Linear Values:
Sum up all the linear values and divide by the number of values to get the mean.
Convert the Mean Linear Value Back to Decibels:
For power ratios:Mean dB=10×log⁡10(Mean Linear Value)Mean dB=10×log10(Mean Linear Value)
For amplitude ratios:Mean dB=20×log⁡10(Mean Linear Value)Mean dB=20×log10(Mean Linear Value)

"""