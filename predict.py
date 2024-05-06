from params_preproc import *
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
        # print('MAX',(np.max(y_pred - y_true)))
        # print('flatten',(y_pred - y_true).flatten())
        y_pred_denor = denormalize_params(y_pred)
        # print(f"y_pred_denor {debugger_details()}  = ")
        y_true_denor = denormalize_params(y_true)

        # TODO de umblat la float precision ca se incaleca xticks
        # TODO verif de ce apar 22 figuri in loc de 11
        hist_errors(y_pred_denor, y_true_denor, dict_params_order, model_name)


        #TODO update hist errors with the following ideas/packages:
        # https://www.analyticsvidhya.com/blog/2021/10/interactive-plots-in-python-with-plotly-a-complete-guide/
        # https://www.datacamp.com/tutorial/create-histogram-plotly
        # https://plotly.com/python/renderers/

        # TODO implement metrics as well
            #for instance, histogram of errors for each parameter or for entire test set
            #or most frequent error for each parameter and its frequency. maybe also for entire test set
            # average of the difference between the actual value and the predicted one

            # also r2 score

        # TODO implement the following functions
        # if save_results:
        #     saveresults(respath, balance_classes, perspeaker, vowels, intonations, shuffle_mode, variance_type)

