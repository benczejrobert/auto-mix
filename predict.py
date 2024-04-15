from utils import *
def run_predict(path_test, inst_feature_scalers, # save_results, respath,
            latest_model = True):
    for current_folderpath, dirs, files in os.walk(path_test):
        if not len(files):
            continue
        scaler = inst_feature_scalers.scalers[os.path.split(current_folderpath)[-1]]  # get scaler for current channel
        channel_folder = current_folderpath.split(os.sep)[-1]
        path_model = os.path.join('..', 'Model',channel_folder)
        print(f"{debugger_details()} path_model = ",path_model)
        if latest_model:
            model_name = sorted(os.listdir(path_model))[-1]
            path_model = os.path.join(path_model, model_name)
            print(f"latest path_model {debugger_details()} latest path_model = ",path_model)
        else:
            raise Exception(f"{debugger_details()} Not implemented yet")
            # TODO implement a functionality that allows loading model for a specific date if it exists
        model = tf.keras.models.load_model(path_model)
        model.summary()
        # window_length, overlap, non_windowed_db all None, remove them
        x_test, y_true = create_test_npy(os.path.join(path_test, channel_folder), scaler)
        y_pred = model.predict(x_test)
        print(f"{debugger_details()} y_pred = ",y_pred)
        print(f"{debugger_details()} y_true = ",y_true)
        print('MAX',(np.max(y_pred - y_true)))
        # TODO unscale y_pred

        # TODO unscale y_true

        # TODO implement the following functions
        # if save_results:
        #     saveresults(respath, balance_classes, perspeaker, vowels, intonations, shuffle_mode, variance_type)

