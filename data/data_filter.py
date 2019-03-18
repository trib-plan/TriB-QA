
# -*- coding:utf-8 -*-
import json


import json

f = open("./search.train.json", "r", encoding='utf-8')
print(f)


  # print(f.readlines())
yes_no = {}
for each in f.readlines():
    json_dicts = json.loads(each)
    if json_dicts["question_type"] == "YES_NO":
        pos = 0
        # print("yesno_answers = " + str(json_dicts["yesno_answers"]))
        for each_answer in json_dicts["answers"]:
            tmp = {}
            tmp["yesno_answers"] =json_dicts["yesno_answers"][pos]
            tmp["segmented_answers"]=json_dicts["segmented_answers"][pos]
            yes_no[each_answer] = tmp
            # print("fake_answers = " + str(each_answer) + "   yesno_answers = " + str(json_dicts["yesno_answers"][pos]) + "\r", end='')
            pos += 1

    # print(str0)
    # print(json_dicts.items()[0].keys())
    # new_dict = json.loads(f)
    # print(json_dicts)
str0 = json.dumps(yes_no, ensure_ascii=False)
fout = open("finished.json", 'w', encoding='utf-8')
fout.write(str0)
fout.close()
