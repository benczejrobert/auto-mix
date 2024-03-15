from utils import *


# https://pypi.org/project/yodel/
# https://codeandsound.wordpress.com/2014/10/09/parametric-eq-in-python/
# https://github.com/topics/equalizer?l=python

# TODO maybe put sound file into self, as well as the data loader

# TODO make another class for the feature extraction because such a class wouldn't necessarily be project specific HERE

# input_sig <-> output&metadata

# noinspection PyMethodMayBeStatic,PyAttributeOutsideInit
class SignalProcessor:
    """
        This class is used to process signals with various filters and save the processed signals with metadata.
    """
    def __init__(self, in_signal_path, out_signals_root_folder=r"../processed-audio-latest", resample_to=None):

        # TODO in_signal_path and out_signals_root_folder should be
        #  declared somewhere globally like in a class of some sort.

        """
        This function creates a SignalProcessor class instance for a single signal.

        :param in_signal_path:
        :param resample_to:
        @BR20240309 Added the rate parameter and self signal to the class.
        @BR20240313 Added the out_signals_root_folder parameter to the class.
        """
        # TODO maybe also add an output name for the signal idk - like use os.path.split(signal_path)[-1] and
        #  add something b4 the extension

        # TODO maybe modify these values
        # normalization values
        self.dbgain_min = -40
        self.dbgain_max = 40
        self.freq_min = 20
        self.freq_max = 20000
        self.resonance_min = 0
        self.resonance_max = 10

        # paths
        self.in_signal_path = in_signal_path
        self.out_signals_root_folder = out_signals_root_folder

        # signal and rate
        self.signal, self.rate = librosa.load(in_signal_path, sr=resample_to)
        # create filters list basically reset the filters
        self.reset()

        # create the output folder if it doesn't exist
        if not os.path.exists(self.out_signals_root_folder):
            os.mkdir(self.out_signals_root_folder)
        if self.rate is None:
            # TODO make this output current error line and also name of the instance
            raise Exception(f"In class {type(self).__name__}, sample rate was not defined for the current instance.")
        # self.rate = rate # left commented maybe for future use


    @staticmethod
    def write_metadata(file_path, tag, tagval, reset_tags=False, verbose_start=False, verbose_final=False):
        """
        This function writes custom metadata to a file. Is used to write the processing settings to a signal.
        :param file_path:
        :param tag:
        :param tagval:
        :param reset_tags:
        :param verbose_start:
        :param verbose_final:
        :return:

        # TODO This function gives the correctly ordered metadata to the file, with the metadata names in lowercase BUT
           the metadata library writes the metadata names in uppercase AND alphabetic order.
           This behavior can be seen when reading the metadata from the file
           (either with verbose_start, verbose_final or with read_metadata() function).
        """

        with taglib.File(file_path, save_on_exit=True) as song:
            if verbose_start:
                print(f"For file at {file_path}, starting tags are:", song.tags)
            if reset_tags:
                song.tags = dict()  # reset tags
            song.tags[tag] = [tagval]  # set new tags
            if verbose_final:
                print(f"For file at {file_path}, final tags are:", song.tags)

    @staticmethod
    def read_metadata(file_path, verbose=False):
        """
        This function reads the metadata from a file and returns it as a dictionary.
        :param file_path:
        :param verbose:
        :return:
        """
        with taglib.File(file_path) as song:
            # order of retrieved metadata seems to be alphabetic.
            if verbose:
                print(f"For file at {file_path}, tags are:", song.tags)
            return song.tags

    def reset(self):
        self.filters = []

    ###############################>> create_end_to_end_all_proc_vars_combinations <<##################################

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
        This function takes a dict of all possible filter settings and unravels it into a dict of all possible
        combinations of filter settings.

        Example:  dict_all_filter_settings_ranges =
        { "high_pass": {"cutoff": range(100, 101, 1000), "resonance": range(2, 3)},
        "low_shelf": {"cutoff": range(200, 201, 1000), "resonance": range(2, 3), "dbgain": list(range(-12, 13, 12))},
        "peak1": {"center": range(8000, 12001, 1000), "resonance": range(2, 3), "dbgain": list(range(-12, 13, 12))},
        "peak2": {"center": range(1000, 1001), "resonance": range(2, 3), "dbgain": [-40]},
        "low_pass": {"cutoff": range(10000, 10001, 1000), "resonance": range(2, 3)},
        "high_shelf": {"cutoff": [9000], "resonance": [2], "dbgain": [6]}
        }

        :param dict_all_filter_settings_ranges:
        :return:
        example: {'high_pass': [{'cutoff': 100, 'resonance': 2}],
        'low_shelf': [{'cutoff': 200, 'resonance': 2, 'dbgain': -12},
                    {'cutoff': 200, 'resonance': 2, 'dbgain': 0},
                    {'cutoff': 200, 'resonance': 2, 'dbgain': 12}],
         'peak1': [{'center': 8000, 'resonance': 2, 'dbgain': -12},
                   {'center': 8000, 'resonance': 2, 'dbgain': 0},
                   {'center': 8000, 'resonance': 2, 'dbgain': 12}],
         'peak2': [{'center': 1000, 'resonance': 2, 'dbgain': -40}],
         ...
         }
        """
        dict_unraveled_all_filter_settings: dict[Any, list[dict[Any, Any]]] = {}

        for filter_type in dict_all_filter_settings_ranges:
            out_list = self._create_proc_vars_single_filter_type_name(dict_all_filter_settings_ranges[filter_type])
            dict_unraveled_all_filter_settings[filter_type] = out_list
        return dict_unraveled_all_filter_settings

    def _create_all_proc_vars_combinations(self, dict_proc_vars_multiple_filter_type_names, capv_root_filename,
                                           start_index=0,
                                           end_index=None,
                                           cr_proc_number_of_filters=None):

        # TODO check if combinations contain all the 5 filters and no repeated filters - it seems that it does
        #  contain all the filters - but i will keep this to do for further testing @BR20240311

        """
        This function takes a dict of all possible filter settings and creates all possible combinations of them.
        One file name corresponds to a proc var.
        :param cr_proc_number_of_filters: If not None, this restricts the possible combinations
                                    to the subset containing exactly number_of_filters filters
        :param dict_proc_vars_multiple_filter_type_names:  dict of all possible filter settings - values for
                                    the filter parameters along with the filter type names
                                    example: {'high_pass': [{'cutoff': 100, 'resonance': 2}],
                                              'low_shelf': [{'cutoff': 200, 'resonance': 2, 'dbgain': -12},
                                                          {'cutoff': 200, 'resonance': 2, 'dbgain': 0},
                                                          {'cutoff': 200, 'resonance': 2, 'dbgain': 12}],
                                               'peak1': [{'center': 8000, 'resonance': 2, 'dbgain': -12},
                                                         {'center': 8000, 'resonance': 2, 'dbgain': 0},
                                                         {'center': 8000, 'resonance': 2, 'dbgain': 12}],
                                               'peak2': [{'center': 1000, 'resonance': 2, 'dbgain': -40}],
                                               ...
                                              }
        :param capv_root_filename: the name of the file that will be saved after processing
        :param start_index: the start index of the file and of the processing variant (can be used for parallelization)
        :param end_index: the end index of the file and of the processing variant (can be used for parallelization)
        :return: dict of all possible combinations of filter settings for multiple output file names
        {'fname_[no].wav': {'filter_type(s)': {'param(s)': value(s), ...}, ...}, ...}
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
                        # TODO this^ "if" sometimes removes valid combinations - falsely rejects combinations
                        #  that are NOT YET in the output - IDK if this "to do" was addressed or what it was about
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
        for i in range(start_index, end_index):   # TODO maybe add trailing zeros to the index - like 0001, 0002, etc
            dict_all_proc_vars_combinations[f"{capv_root_filename}_{start_index + i}.wav"] = (
                list_all_proc_vars_combinations)[i - start_index]
        return dict_all_proc_vars_combinations

    def _check_dict_param_names_and_ranges(self, dict_param_names_and_ranges):

        # TODO ask Moro for these values
        #  These values should be declared somewhere globally like in a class of some sort.

        """
        This function checks the params are in the correct range.

        cutoff - freq (should be between 20 and 20000)
        center - freq (should be between 20 and 20000)
        resonance - number, should be between [... and ...]
        dbgain
        :param dict_param_names_and_ranges:
        :return:
        """
        if not isinstance(dict_param_names_and_ranges, dict):
            raise Exception(f"dict_param_names_and_ranges is not a dict: {dict_param_names_and_ranges}")
        if len(dict_param_names_and_ranges) < 1:
            raise Exception(f"dict_param_names_and_ranges has no keys: {dict_param_names_and_ranges}")

        for filter_type in dict_param_names_and_ranges:
            for param_name in dict_param_names_and_ranges[filter_type]:
                param_range = dict_param_names_and_ranges[filter_type][param_name]
                param_range = sorted(list(param_range))
                if param_name not in ["cutoff", "center", "resonance", "dbgain"]:
                    raise Exception(f"param_name {param_name} is not a valid filter parameter name. dict is",
                                    dict_param_names_and_ranges)
                if param_name in ["cutoff", "center"]:
                    if param_range[0] < self.freq_min or param_range[-1] > self.freq_max:
                        raise Exception(f"param_range for {param_name} is not in the range [20, 20000]: {param_range}."
                                        f" dict is", dict_param_names_and_ranges)
                if param_name in ["resonance"]:
                    if param_range[0] < self.resonance_min or param_range[-1] > self.resonance_max:
                        raise Exception(f"param_range for {param_name} is not in the range [0, 10]: {param_range}."
                                        f" dict is", dict_param_names_and_ranges)
                if param_name in ["dbgain"]:
                    if param_range[0] < self.dbgain_min or param_range[-1] > self.dbgain_max:
                        raise Exception(f"param_range for {param_name} is not in the range [-40, 40]: {param_range}."
                                        f" dict is", dict_param_names_and_ranges)

    def _number_filter_bands(self, dict_in):  # for create_end_to_end_all_proc_vars_combinations()
        """
        This function numbers the frequency bands in the input dictionary, so they remain ordered in the metadata.
        :param dict_in:
        :return:
        """

        dict_out = {}
        for k in dict_in:
            dict_out[k.lower()] = dict_in[k]
        crt_no = 0
        no_digits = np.log10(len(dict_in)) + 1
        for dk in dict_in:
            if dk in dict_out.keys():
                dict_out.pop(dk)
            print(f"{crt_no + 1}_{dk}".zfill(len(dk)+1+int(no_digits)))
            dict_out[f"{crt_no + 1}_{dk}".zfill(len(dk)+1+int(no_digits))] = dict_in[dk]
            crt_no += 1
        return dict_out

    ###############################>> create_end_to_end_all_proc_vars_combinations <<##################################

    ######################################>> process_signal_all_variants <<############################################

    def _remove_numbers_from_proc_var(self, dict_in):  # for process_signal_all_variants() and load_data_for_training()
        # TODO testme
        """
        This function removes the numbers from the filter band names.
        :param dict_in:
        :return:
        """
        dict_out = {}
        for k in dict_in:
            dict_out[re.sub("[1-9]+", "", k)[1::]] = dict_in[k]
        return dict_out

    def _lowercase_filter_bands(self, dict_in):  # for process_signal_all_variants() and load_data_for_training()
        """
        This function transforms the metadata to lowercase.
        :param dict_in:
        :return:
        """
        dict_out = {}
        for k in dict_in:
            dict_out[k.lower()] = dict_in[k]
        return dict_out

    def _create_filter(self, filter_type_name, dict_params):
        """
        This function creates a filter of the specified type and with the specified parameters.
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
        This function creates all the filters for a single processing variant.
        :param dict_procs_single_variant:

        Example: {low_shelf: {"cutoff": 1000, "resonance": 2.0, "dbgain": 2.0},
                 high_shelf: {"cutoff": 1000, "resonance": 2.0, "dbgain": 2.0}
                 }
        :return:
        """
        # output: np.ndarray[Any, np.dtype[np.floating[np._typing._64Bit] | np.float_]] = np.zeros(signal_in.size)
        dict_procs_single_variant = self._lowercase_filter_bands(dict_procs_single_variant)  # TODO testme
        dict_procs_single_variant = self._remove_numbers_from_proc_var(dict_procs_single_variant)  # todo testme
        print('363 line',dict_procs_single_variant)
        for crt_filter_type_name in dict_procs_single_variant.keys():
            cftn = crt_filter_type_name
            if cftn[-1] in '1234567890':
                cftn = cftn[:-1]
            self._create_filter(filter_type_name=cftn, dict_params={"samplerate": self.rate,
                                                                    **dict_procs_single_variant[
                                                                        crt_filter_type_name]})
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
        for f in self.filters:
            f.process(signal_out, signal_out)
        self.reset()  # after signal variant was processed, reset
        return signal_out
    ######################################>> process_signal_all_variants <<############################################

    #########################################>> load_data_for_training <<##############################################
    def _normalize_value(self, x, p_min, p_max, norm_type='0,1'):
        """

        :rtype: numeric or np.array
        """
        if norm_type == '0,1':
            return (x - p_min) / abs(p_max - p_min)
        if norm_type == '-1,1':
            return (x - p_min) / abs(p_max - p_min) * 2 - 1
        if norm_type == '-1, 1 max abs':
            return x / p_max(np.abs([p_min, p_max]))

    def _normalize_params(self, dict_params, normalize_type='0,1'):
        """
            This function normalizes the parameters in dict_params to the range [0, 1] or [-1,1] based on user input.
            The dict_params is the dict saved in the metadata of the processed signal.

            :param dict_params: dict of filter parameters
            :param normalize_type: '0,1' or '-1,1' or '-1, 1 max abs'
            example:
                {'HIGH_PASS': ["{'cutoff': 100, 'resonance': 2}"],
                'LOW_PASS': ["{'cutoff': 10000, 'resonance': 2}"],
                'LOW_SHELF': ["{'cutoff': 200, 'resonance': 2, 'dbgain': 10}"],
                 'PEAK1': ["{'center': 11000, 'resonance': 2, 'dbgain': 10}"],
                'PEAK2': ["{'center': 1000, 'resonance': 2, 'dbgain': -40}"],
                'HIGH_SHELF': ["{'cutoff': 9000, 'resonance': 2, 'dbgain': 6}"]}
            :return: dict of normalized filter parameters with keys in lowercase
                # TODO or maybe just the normalized parameters in the following order:
                    hp_cutoff, hp_resonance, lp_cutoff, lp_resonance, ls_cutoff, ls_resonance, ls_dbgain,
                     p1_center, p1_resonance, p1_dbgain, p2_center, p2_resonance, p2_dbgain,
                      hs_cutoff, hs_resonance, hs_dbgain - 16 params, 16 output neurons OR without HS and LP - 11 params
        """
        normalized_param = None
        list_out_params = []
        for filter_type in dict_params:
            param_dict = eval(dict_params[filter_type][0])
            for param_name in param_dict:
                # normalize the parameters
                if param_name in ["cutoff", "center"]:
                    normalized_param = self._normalize_value(param_dict[param_name], self.freq_min,
                                                             self.freq_max, norm_type=normalize_type)
                if param_name in ["resonance"]:
                    normalized_param = self._normalize_value(param_dict[param_name], self.resonance_min,
                                                             self.resonance_max, norm_type=normalize_type)
                if param_name in ["dbgain"]:
                    normalized_param = self._normalize_value(param_dict[param_name], self.dbgain_min,
                                                             self.dbgain_max, norm_type=normalize_type)

        # # normalize 0,1
        # normalized_param = (param_dict[param_name] - self.dbgain_min) / (self.dbgain_max - self.dbgain_min)
        # # normalize -1,1
        # normalized_param = (param_dict[param_name] - self.dbgain_min) / (self.dbgain_max - self.dbgain_min) * 2 - 1
        # # normalize -1, 1 max abs
        # normalized_param = param_dict[param_name] / max(np.abs([self.dbgain_min, self.dbgain_max]))
                if normalized_param:
                    list_out_params.append(normalized_param)
        return list_out_params
    #########################################>> load_data_for_training <<##############################################

    def create_end_to_end_all_proc_vars_combinations(self, dict_param_names_and_ranges, root_filename="eq-ed",
                                                     start_index=0, end_index=None, number_of_filters=None,
                                                     sr_threshold=44100):
        """
        This function creates all possible combinations of filter settings and outputs them as a dict of filenames.


        :rtype: dict
        :param dict_param_names_and_ranges: dict of all possible filter settings - values
        example: {
          "high_pass": {"cutoff": range(100, 101, 1000), "resonance": range(2, 3)},
          "low_shelf": {"cutoff": range(200, 201, 1000), "resonance": range(2, 3), "dbgain": list(range(-12, 13, 12))},
          "peak1": {"center": range(8000, 12001, 1000), "resonance": range(2, 3), "dbgain": list(range(-12, 13, 12))},
          "peak2": {"center": range(1000, 1001), "resonance": range(2, 3), "dbgain": [-40]},
          "low_pass": {"cutoff": range(10000, 10001, 1000), "resonance": range(2, 3)},
          "high_shelf": {"cutoff": [9000], "resonance": [2], "dbgain": [6]}
        }
        :param root_filename:
        :param start_index:
        :param end_index:
        :param number_of_filters:
        :param sr_threshold:
        :return:
        """
        self._check_dict_param_names_and_ranges(dict_param_names_and_ranges)
        print(f"{debugger_details()} dict_param_names_and_ranges = ", dict_param_names_and_ranges)
        dict_param_names_and_ranges = self._number_filter_bands(dict_param_names_and_ranges)
        print(f"{debugger_details()} dict_param_names_and_ranges = ", dict_param_names_and_ranges)
        if len(dict_param_names_and_ranges) == number_of_filters:  # TODO test this if
            if self.rate < sr_threshold:  # todo test this
                print(f"Sample rate is below {sr_threshold} Hz, so high shelf and low pass filters will not be used.")
                for filter_type in dict_param_names_and_ranges:
                    if "high_shelf" in filter_type or "low_pass" in filter_type:
                        dict_param_names_and_ranges.pop(filter_type, None)

            dict_unraveled_filter_settings = self._create_proc_vars_multiple_filter_type_names(
                dict_param_names_and_ranges)
            ete_dict_filenames_and_process_variants = self._create_all_proc_vars_combinations(
                dict_unraveled_filter_settings, root_filename, start_index, end_index, number_of_filters)
            return ete_dict_filenames_and_process_variants
        else:  # TODO test this
            raise Exception(f"Number of filters in dict_param_names_and_ranges ({len(dict_param_names_and_ranges)}) "
                            f"does not match the number_of_filters ({number_of_filters}).")


    def process_signal_all_variants(self, asv_dict_filenames_and_process_variants, normalize=True):
        """
        This function processes the input sig with all the processings in the asv_dict_filenames_and_process_variants.
        This function that calls _process_signal_variant multiple times by iterating a dict of dicts.
        # A processing list is a dict of
        { filename(s): {filter_type_name(s): {param_name(s): value(s), ...}, ... }, ... }
        # This function will output the processed signal with a log of all the processing it went through
        # the name of the signal will be like base_name_[index].wav

        :param asv_dict_filenames_and_process_variants:
        :param normalize:
        :return:

        @BR20240309 Added the normalize parameter to the function signature.
        Also removed the asv_signal_in parameter and instead called from self.

        """
        # TODO add an output folder in the self class init and save the files there

        # TODO maybe modify to first call the create_all_proc_vars and then the backtracking idk
        #  IDK if this todo was done or what it was about
        for str_file_pre_extension in asv_dict_filenames_and_process_variants:
            # last e - unprocessed input signal name
            str_unproc_sig_name = self.in_signal_path.split("/")[-1].split(".")[0]
            sig_ext = str_unproc_sig_name.split(".")[-1]  # last e - signal extension
            crt_file_path = r"/".join([self.out_signals_root_folder,
                                       "_".join([str_unproc_sig_name,
                                                str_file_pre_extension+"."+sig_ext])])
            # for filter_type in dict_filenames_and_process_variants[filename]: # added
            np_arr_out_sig = self._process_signal_variant(self.signal, asv_dict_filenames_and_process_variants[
                str_file_pre_extension])  # signal_in -> out_sig
            if normalize:
                np_arr_out_sig = normalization(np_arr_out_sig)
            sf.write(crt_file_path, np_arr_out_sig, self.rate)

            reset = True
            for filter_type in asv_dict_filenames_and_process_variants[str_file_pre_extension]:
                # write signal to disk using metadata as well. write metadata one by one
                self.write_metadata(crt_file_path, filter_type,
                                    str(asv_dict_filenames_and_process_variants[str_file_pre_extension][filter_type]),
                                    reset, False, True)
                reset = False

    def load_data_for_training(self, training_data_folder):

        # TODO create fuction body and use _normalize_params to normalize the parameters in the metadata of
        #  the processed signals and then check how it works

        list_all_metadata = []
        # load metadata
        for file in os.listdir(training_data_folder):
            if file.endswith(".wav"):
                file_path = os.path.join(training_data_folder, file)
                metadata = self.read_metadata(file_path)
                print(f"--- {debugger_details()} metadata for sound file ", file, " ---")
                print("\t", metadata)
                # normalize metadata
                list_normed_params = (self._normalize_params(metadata))
                # it will be like a matrix of params. each row is the list of params for a signal (Y labels)
                list_all_metadata.append(list_normed_params)

        return list_all_metadata  # TODO testme


sig_path = r'D:\PCON\Disertatie\AutoMixMaster\datasets\diverse-test\white-noise-mono.wav'
aas = SignalProcessor(sig_path, resample_to=None)

# Usage tips: You need to add numbers at the end of every signal processing type, because
# you can have multiple of the same type such as peak1, peak2, peak3 etc. - always name them with numbers at the end

# Usage tips: include dbgain 0 if you want to ignore a certain type of filter OR remove it from the below dict.
dict_all_filter_settings = {
    "high_pass": {"cutoff": range(100, 101, 1000), "resonance": range(2, 3)},
    "low_shelf": {"cutoff": range(200, 201, 1000), "resonance": range(2, 3), "dbgain": list(range(-12, 13, 11))},
    "peak1": {"center": range(8000, 12001, 3000), "resonance": range(2, 3), "dbgain": list(range(-12, 13, 11))},
    "peak2": {"center": range(1000, 1001), "resonance": range(2, 3), "dbgain": [-40]},
    "low_pass": {"cutoff": range(10000, 10001, 1000), "resonance": range(2, 3)},
    "high_shelf": {"cutoff": [9000], "resonance": [2], "dbgain": [6]}
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
    print(d, '---')
    print(set(dict_filenames_and_process_variants[d].keys()))
    print(len(set(dict_filenames_and_process_variants[d].keys())))
    print(dict_filenames_and_process_variants[d])
asdf
# aas.process_signal_all_variants(dict_filenames_and_process_variants)
# aas.process_signal_all_variants(signal_in, {test_fname: dict_filenames_and_process_variants[test_fname]})
training_data = aas.load_data_for_training("../processed-audio-latest")
# path = r'F:\PCON\Disertatie\AutoMixMaster\datasets\diverse-test\white-noise.wav'
