import time, os
import argparse


# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-o", "--Output", help = "Log file to read", required=True)

# Read arguments from command line
args = parser.parse_args()

# if args.Output:
#     print("Displaying Output as: % s" % args.Output)
    #Set the filename and open the file
filename = args.Output

file = open(filename,'r')

#Find the size of the file and move to the end
st_results = os.stat(filename)
st_size = st_results[6]
file.seek(st_size)

while 1:
    where = file.tell()
    line = file.readline()
    if not line:
        file.seek(where)
    else:
        print(line) # already has newline
    # time.sleep(1e-323)
