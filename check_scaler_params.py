# def check_scaler_params(remaining_list_of_filepaths,
#                         csp_max_abs, csp_scaler, csp_scaler_type, csp_data_path):
#     if csp_max_abs is not None and remaining_list_of_filepaths is not None:
#         csp_scaler.csp_max_abs_ = csp_max_abs
#         print(
#             f" --- compute_csp_scaler(): For channel {os.path.split(csp_data_path)[-1]}, for scaler parameters in ,{f'{csp_scaler_type}_scaler_values.npy'} there are {len(remaining_list_of_filepaths)} files left to parse--- ")
#         if len(remaining_list_of_filepaths) == 0:
#             return None  # TODO this will not work for multidim csp_scalers and will need an update
#     elif remaining_list_of_filepaths is None:
#         print(
#             f" --- compute_csp_scaler(): For channel {os.path.split(csp_data_path)[-1]}, remaining filepaths are not yet saved --- ")
#         test_train_paths = [csp_data_path.replace("Test", "Train"), csp_data_path]
#         if "Train" in csp_data_path:
#             test_train_paths = [csp_data_path, csp_data_path.replace("Train", "Test")]
#         list_of_filepaths = []
#         for path in test_train_paths:  # generate files to load
#             crt_filepaths = [os.path.join(path, file) for file in sorted(os.listdir(path))]
#             list_of_filepaths.extend(crt_filepaths)
#             remaining_list_of_filepaths = list_of_filepaths  # todo check how to treat this - should initially load ALL files for both test and train for csp_scaler computation
#     return remaining_list_of_filepaths, csp_scaler, csp_max_abs

import threading as th
import time
import datetime

from cffi.backend_ctypes import xrange

# init_time = datetime.datetime.now()
# time.sleep(5)
# print("non parallel for loop took: ", datetime.datetime.now() - init_time)
# for i in a:
#     print(i)
#     sleep(1)
# print("non parallel for loop took: ", datetime.datetime.now() - init_time)

# todo call moro
#  todo prod music synth etc

# todo multithread

# pr_lock = th.Lock()


# n_t = 4

threads = []


# a = list(range(2*2*2*3*5*7*11))
# a = list(range(240)) # 120*2 = 240. divide by 40 = 6. divide by 20 = 12
a = list(range(16)) # 120*2 = 240. divide by 40 = 6. divide by 20 = 12
# a = list(range(20)) # 120*2 = 240. divide by 40 = 6. divide by 20 = 12
def parallel_print(list_print):
    for i in list_print:
        print(th.current_thread(), "parprint i=",i)
        time.sleep(1) # simulate operation that takes time
        print("return sleep")
    # return max(list_print)
lvec = len(a)
# 120 sec divided by... 8 threads = 15 sec
# 120 / 20 threads = 6 sec
# 120 / 24 = 5 sec
n_t = 4 # 40 threads = 3s
len_subvec = lvec//n_t


for i in range(n_t):
    crt_rng = ((i)*len_subvec,(i)*n_t+len_subvec)
    crt_vec = a[slice(*crt_rng)]
    # print(crt_rng)
    # print("////i", crt_vec)
    t = th.Thread(target=parallel_print, args=(crt_vec,))
    # print("get thread b4 start",t.getName())
    threads.append(t)

print("-------start code-------")
init_time = datetime.datetime.now()
# init_time = time.thread_time()


for t in threads:
    t.start()
    # t.join()        # why does join have to be in a separate for?
for t in threads: # TODO why does join make such a big difference? why does it make the running time correct but no join makes it appear as if it's 0??
    t.join()
# if you don't join, the main thread will continue and print the time before the threads finish
# join indicates that the main thread should wait for the threads to finish before continuing
# or also the end of a parallel section

# if join is in the same for loop as start, the main thread will wait for each thread to finish before starting the next one



time_duration = datetime.datetime.now() - init_time
print(f"parallel for loop took: , {time_duration}")


import threading

def something():
    ao = [5,1,3,9,7]
    mx = ao[0]
    for i in ao:
        if i > mx:
            mx = i
            yield mx
        print ("Hello")

def my_thing():
    bo = [4, 2,8,6,10]
    mx = bo[0]
    for i in bo:
        if i > mx:
            mx = i
            yield mx
        print ("world")

def get_max_from_inlist(inlist):
    mx = inlist[0]
    for i in inlist:
        if i > mx:
            mx = i
        yield mx

aq = threading.Thread(target=something)
aq.start()
aqb = threading.Thread(target=my_thing)
aqb.start()
aq.join()
aqb.join()