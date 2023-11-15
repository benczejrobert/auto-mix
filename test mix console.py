
from utils import *
# https://pypi.org/project/yodel/
# https://codeandsound.wordpress.com/2014/10/09/parametric-eq-in-python/
# https://github.com/topics/equalizer?l=python


 # with save_on_exit=True, file will be saved at the end of the 'with' block

write_metadata("collab.wav",{"freq":["6900"]})


class signal_processor:

    # TODO add

    def __init__(self, rate = None):
        if rate is None:
            # TODO make this output current error line and also name of the instance
            raise Exception(f"In class {type(self).__name__}, sample rate was not defined for the current instance.")
        self.rate = rate
        self.bi_quad = filter.Biquad()
        # self.eq = filter.




    #TODO make a function that calls process_signal multiple times by iterating a processing_variant
    # A processing_variant is a list of [dict of {proc_type: [params]}]
    # - this function will output the processed signal with a log of all the processing it went through
    # the name of the signal will be like base_name_index


    # TODO make a function that iterates a list of processing variants for a signal. - this will contain a list of processing variants
    def single_process(self, signal, proc_type, params):
        print(params)
        print(*params)

        # TODO try doing multiple processings and use a self.filters list instead.

        # TODO folosesc peak
        try:
            process = getattr(self.bi_quad, proc_type) #(self.rate, params[0], params[1],2))
        except:

            process(self.rate,*params)
        output: np.ndarray[Any, np.dtype[np.floating[np._typing._64Bit] | np.float_]] = np.zeros(signal.size)
        # self.bi_quad.process(self.bi_quad,signal, output)
        """
        :return:
        """
# print(globals().items())
aas = signal_processor(48000)
aas.single_process(np.ones(48000),"low_pass",[100,2])
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
