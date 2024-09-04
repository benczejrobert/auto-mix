import numpy as np

from k_fold import k_fold
from utils import *

def k_fold_cross_validation(path, k, path_model_input, batch_size,
                            epochs, optimizer, dropout, inst_feature_scalers, shuffle_mode):
    '''
    Trains the model using K-fold algorithm
    
    Arguments:
        - path [string], relative path to Train folder
        - k [int], number of subsets used to split the Train set into
        - path_model [string], relative path to save the trained model
        - batch_size [int], the size of a batch used to train the model
        - shuffle_mode [boolean], if False shuffle the data from train and val separately

    @BR20240414 Renamed path_model to path_model_input to avoid overwriting in the path_model variable in the function
    @BR20240415 Renamed model name to include the current date and time and avoid overriting
    '''
    for current_folderpath, dirs, files in os.walk(path):
        if not len(files):
            continue
        scaler = inst_feature_scalers.scalers[os.path.split(current_folderpath)[-1]]  # get scaler for current channel

        channel = os.path.split(current_folderpath)[-1]
        model_folder_path = os.path.join(os.path.split(path_model_input)[0], channel)
        model_name = os.path.split(path_model_input)[-1].split('.')[0]\
                        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5'
        if not os.path.exists(model_folder_path):  # create Model folder if it does not exist
            os.mkdir(model_folder_path)
        path_model = os.path.join(model_folder_path, model_name)
        for i in range(k):
            x_train, x_val, y_train, y_val = k_fold(current_folderpath, k, i + 1)
            x_train = scaler.transform(x_train)
            x_val = scaler.transform(x_val)
            if np.max(x_val) > 1.1 or np.min(x_val) < -1.1 or np.max(x_train) > 1.1 or np.min(x_train) < -1.1:
                raise Exception("Transformed data has been scaled incorrectly")

            if shuffle_mode:    # shuffle data samples from train and val together and then separate them again
                train_length = x_train.shape[0]
                x_data = np.concatenate((x_train, x_val), axis=0)
                y_data = np.concatenate((y_train, y_val), axis=0)
                x_data_shuff, y_data_shuff = shuffle(x_data, y_data, random_state=42)
                x_train = x_data_shuff[0:train_length, :]
                x_val = x_data_shuff[train_length:len(x_data_shuff), :]
                y_train = y_data_shuff[0:train_length, :]
                y_val = y_data_shuff[train_length:len(x_data_shuff), :]
            else:   # shuffle data from train and val separately
                x_train, y_train = shuffle(x_train, y_train, random_state=42)
                x_val, y_val = shuffle(x_val, y_val, random_state=42)


            if len(x_train.shape)<=3:  # i.e. shape is not 3D for conv 1D or not 4D for conv2D  #and not keep_feature_dims or not raw_features
                sh = list(x_train.shape)
                sh.append(1)
                x_train = np.reshape(x_train,tuple(sh))
                sh = list(x_val.shape)
                sh.append(1)
                x_val = np.reshape(x_val,tuple(sh))

            xtr = x_train
            model = create_model(xtr, y_train.shape[-1], optimizer, dropout, summary=True)
            train_model(model, (x_train, y_train), (x_val, y_val), batch_size, epochs, path_model)
            break #only trains for one fold, will be updated