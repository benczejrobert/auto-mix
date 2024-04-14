from utils import *

class ChannelFeatureScalers:
    def __init__(self, trian_data_root, list_scaler_types = None, list_with_mean = None):
        """

        :param trian_data_root:  - relative path to Train folder
        :param list_scaler_types: - list of scaler types to be used for each channel (see compute_scaler() function in utils.py)
        :param list_with_mean:  - list of boolean values indicating whether to use mean for each channel
        """
        self.scalers = {}
        self.train_data_root = trian_data_root
        self.list_scaler_types = list_scaler_types
        self.list_with_mean = list_with_mean
        if list_scaler_types is not None:
            if not len(list_scaler_types) == len(os.listdir(self.train_data_root)):
                raise Exception("The length of list_scaler_types and trian_data_root should be equal")
        else:
            self.list_scaler_types = ['maxabs'] * len(os.listdir(self.train_data_root))
        if list_with_mean is not None:
            if not len(list_with_mean) == len(os.listdir(self.train_data_root)):
                raise Exception("The length of list_with_mean and trian_data_root should be equal")
        else:
            self.list_with_mean = [True] * len(os.listdir(self.train_data_root))

        for current_filepath, dirs, files in os.walk(self.train_data_root):
            i = 0
            if not len(files):
                continue
            channel = os.path.split(current_filepath)[-1]
            self.scalers[channel] = compute_scaler(current_filepath, with_mean=self.list_with_mean[i],
                                                   scaler_type=self.list_scaler_types[i])
            i += 1
