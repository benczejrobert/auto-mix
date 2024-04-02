from utils import *
class SignalProcessor:
    """
        This class is used to process signals with various filters and save the processed signals with metadata.
    """

    def __init__(self, in_signal_path, dict_norm_values, processed_signals_root_folder=r"../processed-audio-latest",
                 features_folder=r"../features-latest", resample_to=None):

        """
        This function creates a SignalProcessor class instance for a single signal.

        :param features_folder:
        :param in_signal_path:
        :param resample_to:
        @BR20240309 Added the rate parameter and self signal to the class.
        @BR20240313 Added the preproc_signals_root_folder parameter to the class.
        """

        self.reset_instance_params(in_signal_path, dict_norm_values, processed_signals_root_folder,
                                   features_folder, resample_to)

    def reset_instance_params(self, in_signal_path, dict_norm_values, processed_signals_root_folder=r"../processed-audio-latest",
                 features_folder=r"../features-latest", resample_to=None):

        """
        This function creates a SignalProcessor class instance for a single signal.

        :param features_folder:
        :param in_signal_path:
        :param resample_to:
        @BR20240309 Added the rate parameter and self signal to the class.
        @BR20240313 Added the preproc_signals_root_folder parameter to the class.
        """

        # TODO ask Moro for these values & modify them
        #  These values should be declared somewhere globally like in a class of some sort.

        # TODO in_signal_path and preproc_signals_root_folder should be
        #  declared somewhere globally like in a class of some sort.
        # normalization values
        for key, value in dict_norm_values.items():
            setattr(self, key, value)

        # paths
        self.in_signal_path = in_signal_path
        self.out_signals_root_folder = processed_signals_root_folder
        self.features_folder = features_folder

        # signal and rate
        self.signal, self.rate = librosa.load(in_signal_path, sr=resample_to)
        # create filters list basically reset the filters
        self.reset_filters()

        # create the output folder if it doesn't exist
        if not os.path.exists(self.out_signals_root_folder):
            os.mkdir(self.out_signals_root_folder)
        if not os.path.exists(self.features_folder):
            os.mkdir(self.features_folder)
        if self.rate is None:
            # TODO make this output current error line and also name of the instance
            #     maybe also line where the class instance is in the code
            raise Exception(f"{debugger_details()} In class {type(self).__name__}, "
                            f"sample rate was not defined for the current instance.")
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

        @BR20240311 This function gives the correctly ordered metadata to the file, with
           the metadata names in lowercase BUT
           the metadata library writes the metadata names in uppercase AND alphabetic order.
           This behavior can be seen when reading the metadata from the file
           (either with verbose_start, verbose_final or with read_metadata() function).

        @BR20240316 This behavior was fixed using the _number_filter_bands() function and _lowercase_filter_bands().
           This fix will not be applied to the verbose_start and verbose_final sections in this function.
        @BR20240319 This function can not modify the metadata of npy files and probably other formats.
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

    def reset_filters(self):
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
        # print("line 38 dict_param_names_and_ranges = ", dict_param_names_and_ranges)
        # print("line 39 values = ", *values)
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

        @BR20240311 checked if combinations contain all the 5 filters and no repeated filters - it seems that it does
        contain all the filters - might need future testing
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
        for i in range(start_index, end_index):  # TODO maybe add trailing zeros to the index - like 0001, 0002, etc
            dict_all_proc_vars_combinations[f"{capv_root_filename}_{start_index + i}.wav"] = (
                list_all_proc_vars_combinations)[i - start_index]
        return dict_all_proc_vars_combinations

    def _check_dict_param_names_and_ranges(self, dict_param_names_and_ranges):

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
        :type dict_in: dict
        :param dict_in:
        :return:
        """

        dict_out = {}
        for k in dict_in:
            dict_out[k.lower()] = dict_in[k]
        crt_no = 0
        no_digits = np.log10(len(dict_in)) + 1
        for dk in dict_in:
            # print(f"{crt_no + 1}_{dk}".zfill(len(dk)+1+int(no_digits)))
            dict_out[f"{crt_no + 1}_{dk}".zfill(len(dk) + 1 + int(no_digits))] = dict_in[dk]
            crt_no += 1
            if dk in dict_out.keys():
                dict_out.pop(dk)
        return dict_out

    ###############################>> create_end_to_end_all_proc_vars_combinations <<##################################

    ######################################>> process_signal_all_variants <<############################################

    def _remove_numbers_from_proc_var(self, dict_in):  # for process_signal_all_variants() and load_data_for_training()
        """
        This function removes the numbers from the filter band names.
        :param dict_in:
        :return:
        @BR20240319 Added ^ and _ to regex to only remove the numbers in the first part of the string.
        Removed the part that trails 1st character.
        """
        dict_out = {}
        for k in dict_in:
            dict_out[re.sub(r"^[1-9]+_", "", k)] = dict_in[k]
            if k in dict_out.keys():
                dict_out.pop(k)
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
            if k in dict_out.keys() and k != k.lower():
                dict_out.pop(k)
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

    def _create_filters_single_proc(self, dict_proc_sg_variant):
        """
        This function creates all the filters for a single processing variant.
        :param dict_proc_sg_variant:

        Example: {low_shelf: {"cutoff": 1000, "resonance": 2.0, "dbgain": 2.0},
                 high_shelf: {"cutoff": 1000, "resonance": 2.0, "dbgain": 2.0}
                 }
        :return:
        """
        # output: np.ndarray[Any, np.dtype[np.floating[np._typing._64Bit] | np.float_]] = np.zeros(signal_in.size)
        dict_proc_sg_variant = self._lowercase_filter_bands(dict_proc_sg_variant)
        dict_proc_sg_variant = self._remove_numbers_from_proc_var(dict_proc_sg_variant)
        for crt_filter_type_name in dict_proc_sg_variant.keys():
            cftn = crt_filter_type_name
            if cftn[-1] in '1234567890':
                cftn = cftn[:-1]
            self._create_filter(filter_type_name=cftn, dict_params={"samplerate": self.rate,
                                                                    **dict_proc_sg_variant[
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
        self.reset_filters()  # after signal variant was processed, reset
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

                # TODO may need to reorder the above params^
                      because the ordering of dict keys was solved : added the numbers to the filter bands
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
                # normalized_param = ((param_dict[param_name] - self.dbgain_min) /
                # (self.dbgain_max - self.dbgain_min) * 2 - 1)
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
        @BR20240316 Modified number of filtexception with debugger info too.
        """
        self._check_dict_param_names_and_ranges(dict_param_names_and_ranges)
        if len(dict_param_names_and_ranges) == number_of_filters:  # TODO may need future testing (20240316 worked fine)
            if self.rate < sr_threshold:
                print(f"Sample rate is below {sr_threshold} Hz, so high shelf and low pass filters will not be used.")
                for filter_type in dict_param_names_and_ranges.copy():
                    if "high_shelf" in filter_type or "low_pass" in filter_type:
                        dict_param_names_and_ranges.pop(filter_type, None)
                        if number_of_filters is not None:
                            number_of_filters -= 1
            # QUESTION/TBD: do I want to number the filter bands before or after removing high freq filters?
            #  before: it's like disabling some bands and leaving others unaffected. after: "renumbering" bands

            # TODO should I also add a safety measure for the filter frequencits (i.e. if there is any freq above sr/2)
                # to raise an exception: "Filter frequency is above the Nyquist frequency."
            dict_param_names_and_ranges = self._number_filter_bands(dict_param_names_and_ranges)
            dict_unraveled_filter_settings = self._create_proc_vars_multiple_filter_type_names(
                dict_param_names_and_ranges)
            ete_dict_filenames_and_process_variants = self._create_all_proc_vars_combinations(
                dict_unraveled_filter_settings, root_filename, start_index, end_index, number_of_filters)
            return ete_dict_filenames_and_process_variants
        else:
            raise Exception(f"{debugger_details()} Number of filters in dict_param_names_and_ranges "
                            f"({len(dict_param_names_and_ranges)}) does not match "
                            f"the number_of_filters ({number_of_filters}).")

    def process_signal_all_variants(self, asv_dict_filenames_and_process_variants, normalize=True, verb_start=False, verb_final=False):
        """
        This function processes the input sig with all the processings in the asv_dict_filenames_and_process_variants.
        This function that calls _process_signal_variant multiple times by iterating a dict of dicts.
        # A processing list is a dict of
        { filename(s): {filter_type_name(s): {param_name(s): value(s), ...}, ... }, ... }
        # This function will output the processed signal with a log of all the processing it went through
        # the name of the signal will be like base_name_[index].wav

        :param asv_dict_filenames_and_process_variants:
        :param normalize:
        :param verb_start:
        :param verb_final:
        :return:

        @BR20240309 Added the normalize parameter to the function signature.
        Also removed the asv_signal_in parameter and instead called from self.
        @BR20240319 Added the verb_start and verb_final parameters to the function signature.
        """

        for str_file_pre_extension in asv_dict_filenames_and_process_variants:
            # last e - unprocessed input signal name
            str_unproc_sig_name = self.in_signal_path.split("/")[-1].split(".")[0]
            # sig_ext not used because .wav is in the so-called pre_extension
            # sig_ext = self.in_signal_path.split("/")[-1].split(".")[-1]  # last e - signal extension
            crt_file_path = r"/".join([self.out_signals_root_folder,
                                       "_".join([str_unproc_sig_name,
                                                 str_file_pre_extension])])
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
                                    reset, verb_start, verb_final)
                reset = False

    def load_labels_metadata_for_training(self, training_data_folder):
        """
        This function loads the metadata from the processed signals and normalizes it based on the param limits
        in the self of  this class.
        :param training_data_folder: [str]
        :return: [list] of [float] with the ordered params for each filter - order as in the metadata

        @BR20240315 Fixed bad normalization for dbgain (it was normalized in -1,1 regardless of normalize type).
        @BR20240316 Tested function
        """

        list_all_metadata = []
        # load metadata
        for crt_file in sorted(os.listdir(training_data_folder)):
            crt_file_path = os.path.join(training_data_folder, crt_file)
            if crt_file_path.endswith(".wav"):
                metadata = self.read_metadata(crt_file_path)
                print(f"--- {debugger_details()} metadata for sound crt_file ", crt_file, " ---")
                print("\t", metadata)
                # normalize metadata
                list_normed_params = (self._normalize_params(metadata))
                # it will be like a matrix of params. each row is the list of params for a signal (Y labels)
                list_all_metadata.append(list_normed_params)

        return list_all_metadata


    def create_features_diff_for_training(self, obj_feature_extractor, processed_audio_folder, pre_diff=True, process_entire_signal = True):
        """
            This function loads the processed signals and extracts the features from them.
            :param obj_feature_extractor: [FeatureExtractor]
            :param processed_audio_folder: [str]
            :param pre_diff: [bool] if True, the function will load the processed signals and subtract the signals to
                                extract the features. Otherwise, it will make a difference between the features.
            :param process_entire_signal: [bool] does not window the input signal in librosa's 2D transformations
            :return: [list] of [np.ndarray] with the features for each signal
        """
        if process_entire_signal:
            obj_feature_extractor.features_dict["hop_length"] = len(self.signal) + 1
        for crt_file in sorted(os.listdir(processed_audio_folder)):
            print(f"--- Creating training features for: {crt_file} ---")
            # load the processed signals
            if crt_file.endswith(".wav"):
                crt_file_path = os.path.join(processed_audio_folder, crt_file)
                # print(f"{debugger_details()} crt_file_path", crt_file_path)
                crt_signal, rate = librosa.load(crt_file_path, sr=self.rate)
                metadata = self.read_metadata(crt_file_path)
                # print(f"{debugger_details()} metadata for sound crt_file", crt_file,':', metadata, " ---")
                list_normed_params = (self._normalize_params(metadata))
                # difference before features extracted
                if pre_diff:
                    output_file_path = os.path.join(self.features_folder, f"features_diff_and_params_{crt_file.split('.')[0]}.npy")
                    # subtract the signals
                    signal_diff = self.signal - crt_signal  # assuming rate was equal for all signals
                    # extract the features
                    diff_features = obj_feature_extractor.extract_features(signal_diff)
                # difference after features extracted
                else:
                    output_file_path = os.path.join(self.features_folder, f"diff_features_and_params_{crt_file.split('.')[0]}.npy")
                    # extract the features
                    features_in_signal = obj_feature_extractor.extract_features(self.signal)
                    features_crt_reference_signal = obj_feature_extractor.extract_features(crt_signal)
                    # subtract the features
                    diff_features = features_in_signal - features_crt_reference_signal
                # print(f"--- {debugger_details()} diff_features shape: {diff_features} ---")
                # print("difference between signals =", self.signal - crt_signal) #TODO 2024-03-29 solve BUG MFCC - not present for cepstrum or fft
                # save the features with the metadata:
                np.save(output_file_path, (diff_features, list_normed_params))  # saves tuple of features and params

        # asdf # intentionally added for code to crash here

