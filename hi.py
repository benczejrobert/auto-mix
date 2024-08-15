from time import sleep
i=0
while 1:
    i+=1
    print("hi im a test", i)
    sleep(0.5)
    if i == 1000:
        raise Exception("Test error")



# slurm job stops when closing connection - WHY?
# job runs slow - WHY?

# multithread the scaler
# lookup howto reopen console of a job
# lookup howto allocate more gpus maybe?

# maybe check if scaler looks for max abs per column or if it updates the entire max abs vector