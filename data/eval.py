# -*- coding：utf-8 -*-
import json
import thulac
import rouge
# 我用了一个清华的中文分词软件。
real_answer = {}
parser = thulac.thulac(seg_only=True)
with open("F:\TriB-QA\data\\f2.search.dev.json", "r", encoding='utf-8') as reader:
    examples = []
    for line in reader:
        example = json.loads(line)
        real_answer[example['id']] = parser.cut(example['fake_answer'][0],text=True)
real_ans = list(real_answer.values())


path =  'C:\\Users\\workstation\\Desktop\\predictions.json'
with open(path, encoding='utf-8') as f:
    data = json.load(f)
prediction  = list(data.values())

cleaned_pre = []
for i in prediction:
    i = i.split()
    string = []
    for a in i:
        if a == ' ':
            continue
        else:
            string.append(a)
    cleaned_pre.append(parser.cut((''.join(string)),text=True))

r = rouge.Rouge()
n = 0
sum = 0.0
for i in range(len(cleaned_pre)):
    if cleaned_pre[i] == "..." or real_ans[i] == "..." or real_ans[i] == ".." or real_ans[i] == ".":
        continue
    # print(i)
    # print(r.get_scores(cleaned_pre[i], real_ans[i]))
    n = n + 1
    a = r.get_scores(cleaned_pre[i], real_ans[i])
    sum += a[0]['rouge-l']['f']

writer = open("F:\TriB-QA\data\\Trial_Prediction", 'w', encoding='utf-8')
for i in range(len(cleaned_pre)):
    print(i)
    writer.write(json.dumps("Prediction:  " + cleaned_pre[i], ensure_ascii=False))
    writer.write('\n')
    writer.write(json.dumps("Answer:  " + real_ans[i], ensure_ascii=False))
    writer.write('\n')
writer.write("RougeL : F1- %f" % sum/n)
writer.close()

