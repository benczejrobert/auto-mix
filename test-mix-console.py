from typing import Dict, Any, List
from utils import *


# https://pypi.org/project/yodel/
# https://codeandsound.wordpress.com/2014/10/09/parametric-eq-in-python/
# https://github.com/topics/equalizer?l=python

# noinspection PyMethodMayBeStatic,PyAttributeOutsideInit
class SignalProcessor:

    def __init__(self, rate=None):
        if rate is None:
            # TODO make this output current error line and also name of the instance
            raise Exception(f"In class {type(self).__name__}, sample rate was not defined for the current instance.")
        self.rate = rate
        self.reset()

    def reset(self):
        self.filters = []

    ###############################>> PROC VARS CREATION <<######################################################

    def _create_proc_vars_single_filter_type_name(self, dict_param_names_and_ranges):
        """
        This function generates all possible combinations for a single filter type and outputs them as a list.

        :param dict_param_names_and_ranges: {filter_param_name: range()}
        :return: dict of all possible outputs {filter_type_name: list [all dicts in specified ranges]}
        """
        keys = dict_param_names_and_ranges.keys()
        values = dict_param_names_and_ranges.values()
        print("line 38 dict_param_names_and_ranges = ", dict_param_names_and_ranges)
        print("line 39 values = ", *values)
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
        dict_unraveled_all_filter_settings: dict[Any, list[dict[Any, Any]]] = {}

        for filter_type in dict_all_filter_settings_ranges:
            out_list = self._create_proc_vars_single_filter_type_name(dict_all_filter_settings_ranges[filter_type])
            dict_unraveled_all_filter_settings[filter_type] = out_list
        return dict_unraveled_all_filter_settings

    def _create_all_proc_vars_combinations(self, dict_proc_vars_multiple_filter_type_names, root_filename,
                                           start_index=0,
                                           end_index=None,
                                           cr_proc_number_of_filters=None):

        # TODO check if combinations contain all the 5 filters and no repeated filters
        """
        This function takes a dict of all possible filter settings and creates all possible combinations of them.
        :param cr_proc_number_of_filters: If not None, this restricts the possible combinations
                                    to the subset containing exactly number_of_filters filters
        :param dict_proc_vars_multiple_filter_type_names:  dict of all possible filter settings - ranges for
                                    the filter parameters along with the filter type names
        :param root_filename: the name of the file that will be saved after processing
        :param start_index: the start index of the file and of the processing variant (can be used for parallelization)
        :param end_index: the end index of the file and of the processing variant (can be used for parallelization)
        :return: dict of all possible combinations of filter settings
        """

        list_all_proc_vars_combinations = []

        def backtrack(depth, input_dict):
            keys = list(input_dict.keys())
            result = []

            def recursive_backtrack(index, current_combination):
                if len(current_combination) == depth:
                    result.append(current_combination.copy())
                    return

                for j in range(index, len(keys)):
                    current_key = keys[j]
                    for element in input_dict[current_key]:
                        # if element not in current_combination.values():
                        # TODO this sometimes removes valid combinations - falsely rejects combinations
                        #  that are NOT YET in the output - IDK if this todo was done or what it was about
                        current_combination[current_key] = element
                        recursive_backtrack(j + 1, current_combination)
                        del current_combination[current_key]

            recursive_backtrack(0, {})

            return result

        in_dict_keys = list(dict_proc_vars_multiple_filter_type_names.keys())
        if cr_proc_number_of_filters is None:
            for crt_depth in range(len(in_dict_keys)):
                output = backtrack(crt_depth + 1, dict_proc_vars_multiple_filter_type_names)
                list_all_proc_vars_combinations.extend(output)
        else:
            output = backtrack(cr_proc_number_of_filters, dict_proc_vars_multiple_filter_type_names)
            list_all_proc_vars_combinations.extend(output)
        dict_all_proc_vars_combinations = {}
        if end_index is None:
            end_index = start_index + len(list_all_proc_vars_combinations)
        for i in range(start_index, end_index):
            dict_all_proc_vars_combinations[f"{root_filename}_{start_index + i}.wav"] = list_all_proc_vars_combinations[
                i - start_index]
        return dict_all_proc_vars_combinations

    def create_end_to_end_all_proc_vars_combinations(self, dict_param_names_and_ranges, root_filename="eq-ed",
                                                     start_index=0, end_index=None, number_of_filters=None,
                                                     sr_threshold=44100):
        if len(dict_param_names_and_ranges) == number_of_filters:  # TODO test this if
            if self.rate < sr_threshold:  # todo test this
                print(f"Sample rate is below {sr_threshold} Hz, so high shelf and low pass filters will not be used.")
                for filter_type in dict_param_names_and_ranges:
                    if "high_shelf" in filter_type or "low_pass" in filter_type:
                        dict_param_names_and_ranges.pop(filter_type, None)

            dict_unraveled_filter_settings = self._create_proc_vars_multiple_filter_type_names(
                dict_param_names_and_ranges)
            ete_dict_filenames_and_process_variants = self._create_all_proc_vars_combinations(
                dict_unraveled_filter_settings,
                root_filename, start_index,
                end_index, number_of_filters)
            return ete_dict_filenames_and_process_variants
        else:  # TODO test this
            raise Exception(f"Number of filters in dict_param_names_and_ranges ({len(dict_param_names_and_ranges)}) "
                            f"does not match the number_of_filters ({number_of_filters}).")

    ############################################################################################################
    def _create_filter(self, filter_type_name, dict_params):
        """
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
            cftn = crt_filter_type_name
            if cftn[-1] in '1234567890':
                cftn = cftn[:-1]
            self._create_filter(filter_type_name=cftn, dict_params={"samplerate": self.rate,
                                                                    **dict_procs_single_variant[crt_filter_type_name]})
        return self.filters

    def _process_signal_variant(self, pv_signal_in, dict_procs_single_variant):
        """
        Outputs the input signal with all the processings in the dict_procs_single_variant
        :param pv_signal_in:
        :param dict_procs_single_variant:
        example:
         {
            "low_shelf": {"cutoff": 1000, "resonance": 2.0, "dbgain": 2.0},
            "high_shelf": {"cutoff": 2000, "resonance": 2.0, "dbgain": 2.0}
         }

        :return:
        """

        self._create_filters_single_proc(dict_procs_single_variant)
        signal_out = pv_signal_in.copy()
        print("line 93 preproc comp in/out", pv_signal_in == signal_out)  # test if the signal was copied
        for f in self.filters:
            # print("crt filter coeffs: a", f._a_coeffs, "b", f._b_coeffs)
            sig_out_pre = signal_out.copy()
            f.process(signal_out, signal_out)
            # time.sleep(0.01) # wait 10 ms
            print("line 97 preproc comp in/out", sig_out_pre == signal_out)  # test if the signal was processed

        self.reset()  # after signal variant was processed, reset
        return signal_out

    def process_signal_all_variants(self, asv_signal_in, asv_dict_filenames_and_process_variants, normalize=True):

        # this function that calls _process_signal_variant multiple times by iterating a dict of dicts.
        # A processing list is a dict of { filename: {filter_type_name: {param_name: value} } }
        # - this function will output the processed signal with a log of all the processing it went through
        # the name of the signal will be like base_name_index

        # TODO maybe modify to first call the create_all_proc_vars and then the backtracking idk
        #  IDK if this todo was done or what it was about
        for filename in asv_dict_filenames_and_process_variants:
            # out_sig = signal_in.copy() #added
            # for filter_type in dict_filenames_and_process_variants[filename]: # added
            out_sig = self._process_signal_variant(asv_signal_in, asv_dict_filenames_and_process_variants[
                filename])  # signal_in -> out_sig
            if normalize:
                out_sig = normalization(out_sig)
            sf.write(filename, out_sig, self.rate)

            reset = True
            for filter_type in asv_dict_filenames_and_process_variants[filename]:
                # write signal to disk using metadata as well. write metadata one by one
                write_metadata(filename, filter_type,
                               str(asv_dict_filenames_and_process_variants[filename][filter_type]),
                               reset, False, True)
                reset = False


# contact@romainclement.com
# signal_in, sr = sf.read(r'D:\PCON\Disertatie\AutoMixMaster\datasets\diverse-test\white-noise.wav')
signal_in, sr = sf.read(r'D:\PCON\Disertatie\AutoMixMaster\datasets\diverse-test\white-noise-mono.wav')
# signal_in, sr = sf.read('eq_ed_20.wav')
# signal_in, sr = sf.read('c_eq_ed_10-20.wav')

aas = SignalProcessor(sr)

# Usage tips: You need to add numbers at the end of every signal processing type, because
# you can have multiple of the same type like peak1, peak2, peak3 etc - always name them with numbers at the end

# Usage tips: include dbgain 0 if you want to ignore a certain type of filter OR remove it from the below dict.
dict_all_filter_settings = {
    "high_pass": {"cutoff": range(1000, 1001, 1000), "resonance": range(2, 3)},
    "low_shelf": {"cutoff": range(1000, 1001, 1000), "resonance": range(2, 3), "dbgain": list(range(0, 25, 12))[1::]},
    "peak1": {"center": range(8000, 12001, 1000), "resonance": range(2, 3), "dbgain": list(range(0, 25, 12))[1::]},
    "peak2": {"center": range(100, 101), "resonance": range(2, 3), "dbgain": [-40]},
    "low_pass": {"cutoff": range(1000, 1001, 1000), "resonance": range(2, 3)},
    "high_shelf": {"cutoff": [1000], "resonance": [2], "dbgain": [2]}
}
# Change this to the number of filters you want to use or None
# to use all possible combinations of filters, any number of filters.
no_filters = len(dict_all_filter_settings)

# aas._create_all_proc_vars_combinations()
dict_filenames_and_process_variants = aas.create_end_to_end_all_proc_vars_combinations(dict_all_filter_settings,
                                                                                       root_filename="eq_ed",
                                                                                       start_index=0, end_index=None,
                                                                                       number_of_filters=no_filters)
for d in dict_filenames_and_process_variants:
    print(d)
    print(dict_filenames_and_process_variants[d])
# asdf

# aas.process_signal_all_variants(signal_in,dict_filenames_and_process_variants)
# test_fname = 'eq_ed_9.wav' # 12 kHz
# test_fname = 'eq_ed_10.wav' # 100 Hz
# test_fname = 'eq_ed_10.wav'
test_fname = 'eq_ed_20.wav'
asdf
aas.process_signal_all_variants(signal_in, {test_fname: dict_filenames_and_process_variants[test_fname]})
print(aas.filters)




path = r'F:\PCON\Disertatie\AutoMixMaster\datasets\diverse-test\white-noise.wav'

"""
    This utility will create variations of a signal based on the input parameters and their values.
    
    Parameters:
        + input signal path
        // sample rate is extracted from the signal
        list of dict like -
         {param_name: list - [Param start, Param stop, Params increment, Params increment scale (log or linear)] }
        ^ this dict is the base, for a single processing. 
        // TODO see how to make multiple processing, like combine 2 dicts. 
        - solution: have multiple param names in a list. 
        // first put simple dicts then put multiple param names in the same dict. 
        if the dict contains multiple parameters, 
            then it means those params will be combined and there will be nested for loops. 
            the number of for loops depends on the number of dict keys. 
            there will be a mandatory "outer" for loop that iterates through all dicts.
            
            // to do all combinations of dicts for the "inner" for loops, backtracking will be required. 
        
        {   Param name is based on the processing type (filter type or comp param) and the param name. 
        for instance, see below:
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
