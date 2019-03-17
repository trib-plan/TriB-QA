# import json
# with open("trial_data.txt",encoding="utf-8") as fmodel:
# 	for line in fmodel:
# 		example  = json.loads(line)
#       	print(example)

import json

with open("trial_data.txt",encoding="utf-8") as reader:
    for line in reader:
        example = json.loads(line)
        print(example)
        break