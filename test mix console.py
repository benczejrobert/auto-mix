
from utils import *
# https://pypi.org/project/yodel/
# https://codeandsound.wordpress.com/2014/10/09/parametric-eq-in-python/
# https://github.com/topics/equalizer?l=python

# write_metadata("collab.wav",{"freq":["6900"]})



class signal_processor:

    # TODO add

    def __init__(self, rate = None):
        if rate is None:
            # TODO make this output current error line and also name of the instance
            raise Exception(f"In class {type(self).__name__}, sample rate was not defined for the current instance.")
        self.rate = rate
        self.reset()

    def reset(self):
        self.filters = []

    def _create_filter(self, filter_type_name, dict_params):
        """

        :param filter:
        :param filter_type_name: one of: low_pass, high_pass, peak, low_shelf, high_shelf
        :param dict_params: IF [filter_type_name] in [low_pass, high_pass] -
                       dict like {
                       "cutoff": [number],
                       "resonance": [number]
                       }
                       With reduction of 12 dB/oct
                       IF [filter_type_name] in [peak, low_shelf, high_shelf] -
                        dict like {
                       "cutoff": [number],
                       "resonance": [number],
                       "dbgain": [number]
                       }

        :return:
        """
        print(filter_type_name, **dict_params)
        new_filter = filter.Biquad()
        new_filter.__getattribute__(filter_type_name)(**{"samplerate": self.rate, **dict_params})
        self.filters.append(new_filter)
        del new_filter
        """
        :return:
        """
    def _create_filters_single_proc(self, dict_procs_single_variant):
        """

        :param dict_procs_single_variant:

        Example: {low_shelf: {"cutoff": 1000, "resonance": 2.0, "dbgain": 2.0},
                 high_shelf: {"cutoff": 1000, "resonance": 2.0, "dbgain": 2.0}
                 }
        :return:
        """
        # output: np.ndarray[Any, np.dtype[np.floating[np._typing._64Bit] | np.float_]] = np.zeros(signal_in.size)
        for crt_filter_type_name in dict_procs_single_variant.keys():
            self._create_filter(filter_type_name=crt_filter_type_name,dict_params={"samplerate":self.rate,
                                                                     **dict_procs_single_variant[crt_filter_type_name]})
        return
    def _process_signal_variant(self,signal_in,dict_procs_single_variant):


        """
        Outputs the input signal with all the processings in the dict_procs_single_variant
        :param signal_in:
        :param dict_procs_single_variant:
        example:
         {
            "low_shelf": {"cutoff": 1000, "resonance": 2.0, "dbgain": 2.0},
            "high_shelf": {"cutoff": 2000, "resonance": 2.0, "dbgain": 2.0}
         }

        :return:
        """

        #TODO make a function that calls _process_signal_variant multiple times by iterating a dict of dicts.
        # A processing list is a dict of { filename: {filter_type_name: {param_name: value} } }
        # - this function will output the processed signal with a log of all the processing it went through
        # the name of the signal will be like base_name_index
        
        self._create_filters_single_proc(dict_procs_single_variant)
        signal_out = signal_in
        for f in self.filters:
            f.process(signal_out,signal_out)
        self.reset() # after signal variant was processed, reset
        return  signal_out
    
    def _create_proc_vars_single_filter_type_name(self, dict_param_names_and_ranges):
        """
        This function generates all possible combinations for a single filter type and outputs them as a list.

        :param dict_param_names_and_ranges: {filter_param_name: range()}
        :return: dict of all possible outputs {filter_type_name: list [all dicts in specified ranges]}
        """
        keys = dict_param_names_and_ranges.keys()
        values = dict_param_names_and_ranges.values()

        combinations = list(product(*values))

        list_param_names_combo = []
        for combo in combinations:
            output_dict = dict(zip(keys, combo))
            list_param_names_combo.append(output_dict)
        return list_param_names_combo

    def _create_proc_vars_multiple_filter_type_names(self, dict_all_filter_settings_ranges):
        """

        :param dict_all_filter_settings_ranges:
        :return:
        """
        dict_unraveled_all_filter_settings = {}

        for filter_type in dict_all_filter_settings_ranges:
            out_list = self._create_proc_vars_single_filter_type_name(dict_all_filter_settings_ranges[filter_type])
            dict_unraveled_all_filter_settings[filter_type] = out_list
        return dict_unraveled_all_filter_settings

    def _create_all_proc_vars_combinations(self,proc_vars_multiple_filter_type_names):

        list_all_proc_vars_combinations = []

        def backtrack(depth, input_dict):
            keys = list(input_dict.keys())
            result = []

            def recursive_backtrack(index, current_combination):
                if len(current_combination) == depth:
                    result.append(current_combination.copy())
                    return

                for i in range(index, len(keys)):
                    current_key = keys[i]
                    for element in input_dict[current_key]:
                        # if element not in current_combination.values(): #TODO this sometimes removes valid combinations - falsely rejects combinations that are NOT YET in the output
                        current_combination[current_key] = element
                        recursive_backtrack(i + 1, current_combination)
                        del current_combination[current_key]

            recursive_backtrack(0, {})

            return result

        in_dict_keys = list(proc_vars_multiple_filter_type_names.keys())
        for depth in range(len(in_dict_keys)):
            output = backtrack(depth + 1, proc_vars_multiple_filter_type_names)
            # print("backtrack output = ", output) #TODO delme
            list_all_proc_vars_combinations.extend(output)
        return list_all_proc_vars_combinations

    def create_end_to_end_all_proc_vars_combinations(self,dict_param_names_and_ranges):

        dict_unraveled_filter_settings = self._create_proc_vars_multiple_filter_type_names(dict_param_names_and_ranges)
        # print("unraveled dict", dict_unraveled_filter_settings)  ### TODO delme
        list_all_process_variants = self._create_all_proc_vars_combinations(dict_unraveled_filter_settings)
        return list_all_process_variants
    def process_signal_all_variants(self,signal_in,dict_filenames_and_process_variants):

        # TODO maybe modify to first call the create_all_proc_vars and then the backtracking idk
        for filename in dict_filenames_and_process_variants:
            out_sig = self._process_signal_variant(signal_in,dict_filenames_and_process_variants[filename])
            # after signal_in was processed, reset
            # write signal to disk using metadata as well

            # TODO add signal filename also
        pass

sr = 48000
aas = signal_processor(sr)

dict_process_variant_test = {
                    "low_shelf": {"cutoff": 1000, "resonance": 2.0, "dbgain": 2.0},
                    "high_shelf": {"cutoff": 2000, "resonance": 2.0, "dbgain": 2.0}
                    }

dict_all_filter_settings = {
    "low_shelf": {"cutoff": range(1000, 2000, 1000), "resonance": range(2, 4), "dbgain": range(2, 3)}, # TODO here just exclude the dbgain = 0 easy.
    "high_shelf": {"cutoff": range(1000, 1001), "resonance": range(2, 3), "dbgain": range(2, 3)}
}
# aas._create_all_proc_vars_combinations()

out_list = aas.create_end_to_end_all_proc_vars_combinations(dict_all_filter_settings)


# TODO backtracking function could clean_input_list() and give an error if one dict contains multiple variants.
#  the input list should only contain SINGLE FILTERS 
#  2) then it could give an index to all of the dicts in the cleaned_input_list 
#  3) backtracking on said indices
#  4) output a dict of {filename_index (or index_comb): process_variant_dict}



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


# save out the processed audio
# wavfile.write("output_yodel.wav",rate,out)
# sf.write(f"output_loshelf{str(gain)}dB_Q10_1k.wav", out, rate) # TODO what to use for writing? librosa, wavfile?

# TODO maybe pymixconsole gave wrong results due to the incorrect read/write of the audio
