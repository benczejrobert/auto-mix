from utils import *

for file in os.listdir("."):
    if file.endswith("wav"):
        tags = read_metadata(file, verbose=False)
        print('   ',tags)
        for tag in tags:
            print('proc = ',tag+',', json.loads(tags[tag][0].replace("'",'"')))
