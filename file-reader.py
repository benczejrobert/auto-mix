from utils import *

path = "." # modify this to point to your dataset
for file in os.listdir(path):
    if file.endswith("wav"):
        tags = read_metadata(file, verbose=False)
        print('   ',file)
        for tag in tags:
            print('proc = ',tag+',', json.loads(tags[tag][0].replace("'",'"')))
