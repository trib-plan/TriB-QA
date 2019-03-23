
# -*- coding:utf-8 -*-
import json


import json

f = open("./search.train.json", "r", encoding='utf-8')
print(f)


  # print(f.readlines())
yes_no_train = {}
yes_no_dev = {}
yes_no_test = {}
readlines = f.readlines()
len0 = 15496
# exit()
total = 0
for each in readlines:
    json_dicts = json.loads(each)
    if json_dicts["question_type"] == "YES_NO":
        pos = 0
        # print("yesno_answers = " + str(json_dicts["yesno_answers"]))
        for each_answer in json_dicts["answers"]:
            tmp = {}
            tmp["yesno_answers"] =json_dicts["yesno_answers"][pos]
            # print(json_dicts["yesno_answers"][pos])
            # if total > 10:
            #     exit()
            if json_dicts["yesno_answers"][pos] == "Yes":
                tmp["yesno_answers"] = 1
            elif json_dicts["yesno_answers"][pos] == "No":
                tmp["yesno_answers"] = 0
            else:
                tmp["yesno_answers"] = 2

            # print(json_dicts)
            # exit()
            tmp["segmented_answers"]=json_dicts["segmented_answers"][pos]
            if "yesno_type" in json_dicts.keys():
                tmp["yesno_type"] = json_dicts["yesno_type"]
            print(str(total) + '\r', end='')
            if total < 0.7 * len0:
                yes_no_train[each_answer] = tmp
            else:
                # print("lala")
                yes_no_dev[each_answer] = tmp
            # print("fake_answers = " + str(each_answer) + "   yesno_answers = " + str(json_dicts["yesno_answers"][pos]) + "\r", end='')
            pos += 1
            total += 1

f = open("./zhidao.train.json", "r", encoding='utf-8')
print(f)
readlines = f.readlines()

for each in readlines:
    json_dicts = json.loads(each)
    if json_dicts["question_type"] == "YES_NO":
        pos = 0
        # print("yesno_answers = " + str(json_dicts["yesno_answers"]))
        for each_answer in json_dicts["answers"]:
            tmp = {}
            tmp["yesno_answers"] =json_dicts["yesno_answers"][pos]
            # print(json_dicts["yesno_answers"][pos])
            # if total > 10:
            #     exit()
            if json_dicts["yesno_answers"][pos] == "Yes":
                tmp["yesno_answers"] = 1
            elif json_dicts["yesno_answers"][pos] == "No":
                tmp["yesno_answers"] = 0
            else:
                tmp["yesno_answers"] = 2

            # print(json_dicts)
            # exit()
            tmp["segmented_answers"]=json_dicts["segmented_answers"][pos]
            if "yesno_type" in json_dicts.keys():
                tmp["yesno_type"] = json_dicts["yesno_type"]
            print(str(total) + '\r', end='')
            yes_no_train[each_answer] = tmp
            # print("fake_answers = " + str(each_answer) + "   yesno_answers = " + str(json_dicts["yesno_answers"][pos]) + "\r", end='')
            pos += 1
            total += 1
    # print(str0)
    # print(json_dicts.items()[0].keys())
    # new_dict = json.loads(f)
    # print(json_dicts)
str0 = json.dumps(yes_no_train, ensure_ascii=False)
fout = open("train.json", 'w', encoding='utf-8')
fout.write(str0)
fout.close()

str0 = json.dumps(yes_no_dev, ensure_ascii=False)
fout = open("dev.json", 'w', encoding='utf-8')
fout.write(str0)
fout.close()

# str0 = json.dumps(yes_no_test, ensure_ascii=False)
# fout = open("test.json", 'w', encoding='utf-8')
# fout.write(str0)
# fout.close()
