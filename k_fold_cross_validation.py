from k_fold import k_fold
from utils import *

def k_fold_cross_validation(path, k, path_model, batch_size,
                            epochs, optimizer, dropout, scaler, shuffle_mode):
    '''
    Trains the model using K-fold algorithm
    
    Arguments:
        - path [string], relative path to Train folder
        - k [int], number of subsets used to split the Train set into
        - path_model [string], relative path to save the trained model
        - batch_size [int], the size of a batch used to train the model
        - shuffle_mode [boolean], if False shuffle the data from train and val separately 
    '''
    if not os.path.exists(os.path.split(path_model)[0]): #create Model folder if it does not exist
        os.mkdir(os.path.split(path_model)[0])
    for i in range(k):
        x_train, x_val, y_train, y_val = k_fold(path, k, i+1)
        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)
        if shuffle_mode:    #shuffle data samples from train and val together and then separate them again
            train_length = x_train.shape[0]
            data = np.concatenate((x_train, x_val), axis=0)
            y_data = np.concatenate((y_train, y_val), axis=0)
            data_shuff, y_data_shuff = shuffle(data, y_data, random_state=42)
            x_train = data_shuff[0:train_length, :]
            x_val = data_shuff[train_length:len(data_shuff),:]
            y_train = y_data_shuff[0:train_length, :]
            y_val = y_data_shuff[train_length:len(data_shuff),:]
        else:   #shuffle data from train and val separately
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