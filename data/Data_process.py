# -*- coding：utf-8 -*-
import json

# def get_pred(yesno,question,q_type,answers,question_id):
#     return{
#         "yesno_answers":yesno,
#         "question": question,
#         "question_type":q_type,
#         "answers":answers,
#         "question_id":question_id
#     }
#
#
# def get_ref(entity,yesno,question,q_type,answers,question_id):
#     return{
#         "entity_answers": entity,
#         "yesno_answers":yesno,
#         "question": question,
#         "question_type":q_type,
#         "answers":answers,
#         "source": "search",
#         "question_id":question_id
#     }
#
# path =  'C:\\Users\\workstation\\Desktop\\predictions.json'
#
# pre_path =  "F:\TriB-QA\data\pred.json"
# ref_path = "F:\TriB-QA\data\\ref.json"
#
#
# with open(path,encoding='utf-8') as f:
#     data = json.load(f)
# id = list(int(i) for i in data.keys())
# prediction = list(data.values())
#
# ref_writer = open(ref_path,'w',encoding='utf-8')
# pre_writer = open(pre_path,'w',encoding='utf-8')
# with open("D:\迅雷下载\\train_preprocessed\\train_preprocessed\\trainset\\search.train.json",encoding = 'utf-8') as reader:
#     for line in reader:
#         example = json.loads(line)
#
#         if int(example['question_id']) not in id :
#             continue
#         q_id = example['question_id']
#         question = example['question']
#         q_type = example['question_type']
#         answers = example['answers']
#         yesno = []
#         entity = [[]]
#         pred = data[str(q_id)]
#         cleaned_pre = [''.join(pred.split())]
#         pre  = get_pred(yesno,question,q_type,cleaned_pre,q_id)
#         pre_writer.write(json.dumps(pre, ensure_ascii=False))
#         pre_writer.write('\n')
#         if q_type == "ENTITY":
#             entity = example['entity_answers']
#         elif q_type == "YES_NO":
#             yesno = example['yesno_answers']
#         else:
#             pass
#         ref = get_ref(entity,yesno,question,q_type,answers,q_id)
#         ref_writer.write(json.dumps(ref,ensure_ascii=False))
#         ref_writer.write('\n')
#
# ref_writer.close()
# pre_writer.close()
#
#
#
#
# def get_pred(question_id,q_type,answers,yesno):
#     return{
#         "question_id": question_id,
#         "question_type": q_type,
#         "answers": answers,
#         "yesno_answers":yesno
#     }
# raw_zhidao_path = "F:\TriB-QA\\test1\zhidao.test1.json"
# raw_search_path = "F:\TriB-QA\\test1\search.test1.json"
#
# zhidao_prediction_path =  'F:\TriB-QA\\test1\\test_zhidao_predictions.json'
# search_prediction_path =  'F:\TriB-QA\\test1\\test_search_predictions.json'
#
# pre_path =  "F:\TriB-QA\\test1\\pred.json"
#
# with open(zhidao_prediction_path,encoding='utf-8') as f:
#     data = json.load(f)
# id = list(int(i) for i in data.keys())
# prediction = list(data.values())
#
# pre_writer = open(pre_path,'w',encoding='utf-8')
#
# with open(raw_zhidao_path,encoding = 'utf-8') as reader:
#     for line in reader:
#         example = json.loads(line)
#
#         if int(example['question_id']) not in id :
#             continue
#         q_id = example['question_id']
#         question = example['question']
#         q_type = example['question_type']
#         yesno = []
#         pred = data[str(q_id)]
#         cleaned_pre = [''.join(pred.split())]
#         pre  = get_pred(q_id,q_type,cleaned_pre,yesno)
#         pre_writer.write(json.dumps(pre, ensure_ascii=False))
#         pre_writer.write('\n')
#
# print('finished ZHIDAO')
#
# with open(search_prediction_path,encoding='utf-8') as f:
#     data = json.load(f)
# id = list(int(i) for i in data.keys())
# prediction = list(data.values())
#
# with open(raw_search_path,encoding = 'utf-8') as reader:
#     for line in reader:
#         example = json.loads(line)
#
#         if int(example['question_id']) not in id :
#             continue
#         q_id = example['question_id']
#         q_type = example['question_type']
#         yesno = []
#         pred = data[str(q_id)]
#         cleaned_pre = [''.join(pred.split())]
#         pre  = get_pred(q_id,q_type,cleaned_pre,yesno)
#         pre_writer.write(json.dumps(pre, ensure_ascii=False))
#         pre_writer.write('\n')
# pre_writer.close()


YES_NO_PATH = "F:\TriB-QA\\test1\output0.txt"
pre_path =  "F:\TriB-QA\\test1\\pred.json"
def get_pred(question_id,q_type,answers,yesno):
    return{
        "question_id": question_id,
        "question_type": q_type,
        "answers": answers,
        "yesno_answers":yesno
    }
YES_NO_QUESTION = {}
with open(YES_NO_PATH,encoding='utf-8') as f:
    for line in f:
        example = json.loads(line)
        YES_NO = example['yesno_answers']
        QID = int(example['question_id'])
        YES_NO_QUESTION[QID] = YES_NO

id = list(YES_NO_QUESTION.keys())

writer  = open("F:\TriB-QA\\test1\\result.json",'w',encoding='utf-8')
with open(pre_path,encoding='utf-8') as f:
    for line in f:
        example = json.loads(line)
        if int(example['question_id']) not in id or example['question_type'] != "YES_NO":
            writer.write(json.dumps(example,ensure_ascii=False))
            writer.write('\n')
            continue
        q_type = example['question_type']
        assert q_type == "YES_NO"
        q_type = "Yes_No"
        q_id = example['question_id']
        answer = example['answers']
        YES_NO_ANSWER =  YES_NO_QUESTION[q_id]
        updated_example = get_pred(q_id,q_type,answer,YES_NO_ANSWER)
        writer.write(json.dumps(updated_example, ensure_ascii=False))
        writer.write('\n')

writer.close()