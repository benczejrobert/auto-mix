
from utils import *
# https://pypi.org/project/yodel/
# https://codeandsound.wordpress.com/2014/10/09/parametric-eq-in-python/
# https://github.com/topics/equalizer?l=python


 # with save_on_exit=True, file will be saved at the end of the 'with' block

# write_metadata("collab.wav",{"freq":["6900"]})


class signal_processor:

    # TODO add

    def __init__(self, rate = None):
        if rate is None:
            # TODO make this output current error line and also name of the instance
            raise Exception(f"In class {type(self).__name__}, sample rate was not defined for the current instance.")
        # if signal is None:
        #     # TODO make this output current error line and also name of the instance
        #     raise Exception(f"In class {type(self).__name__}, signal was not defined for the current instance.")
        self.rate = rate
        # self.signal = signal
        self.reset()

    def reset(self):
        self.filters = []


    # TODO make a function that iterates a list of processing variants for a signal. - this will contain a list of processing variants
    def _create_filter(self, proc_type, dict_params):
        """

        :param filter:
        :param proc_type: one of: low_pass, high_pass, peak, low_shelf, high_shelf
        :param dict_params: IF [proc_type] in [low_pass, high_pass] -
                       dict like {
                       "cutoff": [number],
                       "resonance": [number]
                       }
                       With reduction of 12 dB/oct
                       IF [proc_type] in [peak, low_shelf, high_shelf] -
                        dict like {
                       "cutoff": [number],
                       "resonance": [number],
                       "dbgain": [number]
                       }

        :return:
        """
        print(dict_params)
        print(*dict_params)

        new_filter = filter.Biquad()
        new_filter.__getattribute__(proc_type)(**{"samplerate": self.rate, **dict_params})
        self.filters.append(new_filter)
        del new_filter
        """
        :return:
        """
    def _create_filters_single_proc(self, dict_procs_single_variant):
        """

        :param dict_procs_single_variant:

        Example: {low_shelf: {"samplerate": 48000,"cutoff": 1000, "resonance": 2.0, "dbgain": 2.0},
                 high_shelf: {"samplerate": 48000,"cutoff": 1000, "resonance": 2.0, "dbgain": 2.0}
                 }
        :return:
        """
        # output: np.ndarray[Any, np.dtype[np.floating[np._typing._64Bit] | np.float_]] = np.zeros(signal.size)
        for crt_proc_type in dict_procs_single_variant.keys():
            self._create_filter(proc_type=crt_proc_type,dict_params={"samplerate":self.rate,
                                                                     **dict_procs_single_variant[crt_proc_type]})
        return
    def _process_signal_variant(self,signal,dict_procs_single_variant):

        #TODO make a function that calls _process_signal_variant multiple times by iterating a dict of dicts.
        # A processing list is a dict of { filename: {proc_type: {param_name: value} } }
        # - this function will output the processed signal with a log of all the processing it went through
        # the name of the signal will be like base_name_index
        
        self._create_filters_single_proc(dict_procs_single_variant)
        signal_out = signal
        for f in self.filters:
            f.process(signal_out,signal_out)
        self.reset() # after signal was processed, reset
        # write signal to disk using metadata as well

        # TODO add signal filename also
        return  signal_out

# print(globals().items())
sr = 48000

aas = signal_processor(sr)

# **{"samplerate": 48000,"cutoff": 1000, "resonance": 2.0, "dbgain": 2.0}
dict_params_test = {"low_shelf": {"cutoff": 1000, "resonance": 2.0, "dbgain": 2.0}}
test_sig = np.ones(sr)
test_processed_sig = aas._process_signal_variant(test_sig, dict_params_test)
print(test_processed_sig)
asdf
path = 'F:\PCON\Disertatie\AutoMixMaster\datasets\diverse-test\white-noise.wav'

"""
    This utility will create variations of a signal based on the input parameters and their values.
    
    Parameters:
        + input signal path
        // sample rate is extracted from the signal
        list of dict like - {param_name: list - [Param start, Param stop, Params increment, Params increment scale (log or linear)] }
        ^ this dict is the base, for a single processing. 
        // TODO see how to make multiple processing, like combine 2 dicts. - solution: have multiple param names in a list. 
        // first put simple dicts then put multiple param names in the same dict. if the dict contains multiple parameters, 
            then it means those params will be combined and there will be nested for loops. 
            the number of for loops depends on the number of dict keys. 
            there will be a mandatory "outer" for loop that iterates through all dicts.
            
            // to do all combinations of dicts for the "inner" for loops, backtracking will be required. 
        
        {   Param name is based on the processing type (filter type or comp param) and the param name. for instance, see below
            list of dicts with processings
            [
                + allowed params for low/high pass (seemingly constant db/octave):
                    - freq           - name it lp_freq or hp_freq
                    - Q or resonance - name it lp_q or hp_q 
                    - db/oct???
                + low/high shelf
                    - freq           - l/hs_freq
                    - Q or resonance - l/hs_q
                    - db gain        - l/hs_g
                + bell (can be multiple bands) aka peak from parametric EQ
                    - freq 
                    - Q or resonance
                    - db gain
                
                + allowed params for comp:
                    -
                
                
                
            ]
            
        }
        + Output signal start index
        // TODO will need a calculator for the last signal index
    How it works: 
        + Iterates list of processings
        + Creates a name for the signal like base_filename_index and the list of processings will be saved into a 
          [FORMAT TO BE DECIDED] file next to the file name 
        + Processes signal
    
    Outputs: 
        + processed signal
        + name of processed signal 
        + File/record/database containing processed signals. - ground truth for training NN. 
            // WHAT FORMAT WOULD IT BE BEST? - structured data with a fixed set of columns.  

"""


bell_band = filter.Biquad()
# bell_band.peak(samplerate=rate,center=1000,resonance=10,dbgain=-48)
gain = -48
bell_band.low_shelf(samplerate=None,cutoff=1000,resonance=10,dbgain=gain)


# save out the processed audio
# wavfile.write("output_yodel.wav",rate,out)
# sf.write(f"output_loshelf{str(gain)}dB_Q10_1k.wav", out, rate) # TODO what to use for writing? librosa, wavfile?

# TODO maybe pymixconsole gave wrong results due to the incorrect read/write of the audio
