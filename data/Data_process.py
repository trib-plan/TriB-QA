# -*- coding：utf-8 -*-

#包含中间所有的格式的转换，主要是Bert生成的答案 只有ID和预测，需要更改。
import json

# 只有训练集的时候，我训练了search集，想自己evaluate下，所以要通过这个更改输出的格式，现在木用了
# 从输出的prediction文件，按照ID补全信息生成 pred 和 ref 两个文件。
def fulfil_prediction_train(raw_path,prediction_path,ref_path,pre_path):
    def get_pred(yesno,question,q_type,answers,question_id):
        return{
            "yesno_answers":yesno,
            "question": question,
            "question_type":q_type,
            "answers":answers,
            "question_id":question_id
        }


    def get_ref(entity,yesno,question,q_type,answers,question_id):
        return{
            "entity_answers": entity,
            "yesno_answers":yesno,
            "question": question,
            "question_type":q_type,
            "answers":answers,
            "source": "search",
            "question_id":question_id
        }

    path =  'C:\\Users\\workstation\\Desktop\\predictions.json'
    pre_path =  "F:\TriB-QA\data\pred.json"
    ref_path = "F:\TriB-QA\data\\ref.json"


    with open(path,encoding='utf-8') as f:
        data = json.load(f)
    id = list(int(i) for i in data.keys())

    ref_writer = open(ref_path,'w',encoding='utf-8')
    pre_writer = open(pre_path,'w',encoding='utf-8')
    with open("D:\迅雷下载\\train_preprocessed\\train_preprocessed\\trainset\\search.train.json",encoding = 'utf-8') as reader:
        for line in reader:
            example = json.loads(line)

            if int(example['question_id']) not in id :
                continue
            q_id = example['question_id']
            question = example['question']
            q_type = example['question_type']
            answers = example['answers']
            yesno = []
            entity = [[]]
            pred = data[str(q_id)]
            cleaned_pre = [''.join(pred.split())]
            pre  = get_pred(yesno,question,q_type,cleaned_pre,q_id)
            pre_writer.write(json.dumps(pre, ensure_ascii=False))
            pre_writer.write('\n')
            if q_type == "ENTITY":
                entity = example['entity_answers']
            elif q_type == "YES_NO":
                yesno = example['yesno_answers']
            else:
                pass
            ref = get_ref(entity,yesno,question,q_type,answers,q_id)
            ref_writer.write(json.dumps(ref,ensure_ascii=False))
            ref_writer.write('\n')

    ref_writer.close()
    pre_writer.close()



# Bert预测训练的时候目前是分开的，search 和 zhidao 分开。所以我要将两个合在一起并补足信息。
# 输出的output 文件 需要进一步放到YES No

def fulfil_prediction_test1(raw_zhidao_path="F:\TriB-QA\\test1\zhidao.test1.json",raw_search_path = "F:\TriB-QA\\test1\search.test1.json",pred_zhidao = 'F:\TriB-QA\\test1\\test_zhidao_1_predictions.json',pred_search = 'F:\TriB-QA\\test1\\test_search_1_predictions.json',output="F:\TriB-QA\\test1\\pred_1.json"):

    def get_pred(question_id,q_type,answers,yesno):
        return{
            "question_id": question_id,
            "question_type": q_type,
            "answers": answers,
            "yesno_answers":yesno
        }
    with open(pred_zhidao,encoding='utf-8') as f:
        data = json.load(f)
    id = list(int(i) for i in data.keys())
    pre_writer = open(output,'w',encoding='utf-8')
    with open(raw_zhidao_path,encoding = 'utf-8') as reader:
        for line in reader:
            example = json.loads(line)

            if int(example['question_id']) not in id :
                continue
            q_id = example['question_id']
            q_type = example['question_type']
            yesno = []
            pred = data[str(q_id)]
            cleaned_pre = [''.join(pred.split())]
            pre  = get_pred(q_id,q_type,cleaned_pre,yesno)
            pre_writer.write(json.dumps(pre, ensure_ascii=False))
            pre_writer.write('\n')
    print('finished ZHIDAO')
    with open(pred_search,encoding='utf-8') as f:
        data = json.load(f)
    id = list(int(i) for i in data.keys())

    with open(raw_search_path,encoding = 'utf-8') as reader:
        for line in reader:
            example = json.loads(line)

            if int(example['question_id']) not in id :
                continue
            q_id = example['question_id']
            q_type = example['question_type']
            yesno = []
            pred = data[str(q_id)]
            cleaned_pre = [''.join(pred.split())]
            pre  = get_pred(q_id,q_type,cleaned_pre,yesno)
            pre_writer.write(json.dumps(pre, ensure_ascii=False))
            pre_writer.write('\n')
    pre_writer.close()


# 把上面的outpu 中YESno 提取出来。号输入到分类BERT
def YES_NO_filter():
    f = open("F:\TriB-QA\\test1\pred1.json", "r", encoding='utf-8')
    yes_no_dev = {}
    readlines = f.readlines()

    total = 0

    for each in readlines:
        json_dicts = json.loads(each)
        if json_dicts["question_type"] == "YES_NO":
            pos = 0

            for each_answer in json_dicts["answers"]:
                tmp = {}

                tmp["question_id"] = json_dicts["question_id"]
                tmp["answer"] = each_answer
                tmp["question_type"] = json_dicts["question_type"]
                tmp["yesno_answers"] = 2

                if "yesno_type" in json_dicts.keys():
                    tmp["yesno_type"] = json_dicts["yesno_type"]
                print(str(total) + '\r', end='')
                yes_no_dev[json_dicts["question_id"]] = tmp

                pos += 1
                total += 1

    str0 = json.dumps(yes_no_dev, ensure_ascii=False)
    fout = open("F:\TriB-QA\\test1\dev1.json", 'w', encoding='utf-8')
    fout.write(str0)
    fout.close()


# YES NO判断完毕也有一个文件，将这个文件与前面文件合并
# 最后输出result文件
def Build_final_result(Yes_No="F:\TriB-QA\\test1\output0.txt",pre_path="F:\TriB-QA\\test1\\pred1.json",result="F:\TriB-QA\\test1\\result1.json"):

    def get_pred(question_id,q_type,answers,yesno):
        return{
            "question_id": question_id,
            "question_type": q_type,
            "answers": answers,
            "yesno_answers":yesno
        }
    YES_NO_QUESTION = {}
    with open(Yes_No,encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            YES_NO = example['yesno_answers']
            QID = int(example['question_id'])
            YES_NO_QUESTION[QID] = YES_NO

    id = list(YES_NO_QUESTION.keys())

    writer  = open(result,'w',encoding='utf-8')
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


def build_answer_rank_dev(search_pre="F:\\baidu_raw_data\search_nbest_predictions.json",zhidao_pre="F:\\baidu_raw_data\zhidao_nbest_predictions.json",search_dev="F:\\baidu_raw_data\dev_preprocessed\dev_preprocessed\devset\\search.dev.json",zhidao_dev="F:\\baidu_raw_data\dev_preprocessed\dev_preprocessed\devset\zhidao.dev.json"):

    out_path = "F:\\baidu_raw_data\dev_answer_rerank_train.json"
    out_path1 = "F:\\baidu_raw_data\dev_answer_rerank_test.json"
    cnt = 0
    with open(search_pre,encoding = 'utf-8') as f:
        data = json.load(f)
    ids = list(int(i) for i in data.keys())
    writer = open(out_path,'w',encoding='utf-8')
    writer1 = open(out_path1, 'w', encoding='utf-8')
    with open(search_dev,encoding='utf-8') as reader:
        for line in reader:
            example = json.loads(line)
            q_id = example['question_id']
            q_type = example['question_type']
            question = example['question']
            answers = data[str(q_id)]
            example = {
                "source": "search",
                "question_id": q_id,
                "question_type": q_type,
                "question":question,
                "answers": answers,
            }
            if cnt % 10 ==0 :
                writer1.write(json.dumps(example, ensure_ascii=False))
                writer1.write('\n')
            else:
                writer.write(json.dumps(example,ensure_ascii=False))
                writer.write('\n')
            cnt+=1
    with open(zhidao_pre, encoding='utf-8') as f:
        data = json.load(f)
    ids = list(int(i) for i in data.keys())
    with open(zhidao_dev, encoding='utf-8') as reader:
        for line in reader:
            example = json.loads(line)
            q_id = example['question_id']
            q_type = example['question_type']
            question = example['question']
            answers = data[str(q_id)]
            example = {
                "source": "zhidao",
                "question_id": q_id,
                "question_type": q_type,
                "question": question,
                "answers": answers
            }
            if cnt % 10 == 0:
                writer1.write(json.dumps(example, ensure_ascii=False))
                writer1.write('\n')
            else:
                writer.write(json.dumps(example, ensure_ascii=False))
                writer.write('\n')
            cnt+=1
    writer.close()
    writer1.close()

def build_baidu_train_v2(source_zhidao="F:\\baidu_raw_data\\train_preprocessed\\train_preprocessed\\trainset\\zhidao.train.json",
                         source_search="F:\\baidu_raw_data\\train_preprocessed\\train_preprocessed\\trainset\\search.train.json",
                         lhy_zhidao="F:\Baidu_train\lhy_train_zhidao.json",
                         lhy_search="F:\Baidu_train\lhy_train_search.json",
                         output="F:\Baidu_train\\baidu_train_v22.json"):

#     question, question_id, doc_tokens, fake_answer, answer_span,is_impossible:true,false
    def get_train(question,question_id,doc_tokens,fake_answer,answer_span,is_impossible,source):
        return{
            "source": source,
            "question":question,
            "question_id": question_id,
            "doc_tokens":doc_tokens,
            "fake_answer": fake_answer,
            "answer_span":answer_span,
            "is_impossible":is_impossible,
        }
    writer = open(output,"w",encoding='utf-8')
    zhidao_dict = {}
    id = []
    with open(lhy_zhidao,"r", encoding='utf-8') as reader:
        for line in reader:
            example = json.loads(line)
            qas_id = example['question_id']
            zhidao_dict[int(qas_id)] = {"question":example['question'],
                                        "doc_tokens":example['doc_tokens'],
                                        "answer_span":example['answer_span'],
                                        "fake_answer":example['fake_answer']
                                        }
    id = list(zhidao_dict.keys())


    all_count = 0
    no_id = 0
    one_doc = 0
    useful_count = 0
    no_doc =0
    with open(source_zhidao,'r', encoding='utf-8') as reader:
        for line in reader:
            example = json.loads(line)
            all_count +=1
            if int(example['question_id']) not in id:
                no_id+=1
                continue
            try:
                assert len(example["documents"]) > 1
            except AssertionError:
                one_doc+=1
                # print(q_id)
                continue

            q_id = int(example['question_id'])
            question = example['question']
            fake_answer = example['fake_answers']
            assert fake_answer == zhidao_dict[q_id]['fake_answer']

            ans_doc = int(example['answer_docs'][0])
            for i in range(len(example['documents'])-1,-1,-1):
                doc_tokens = []
                if i == ans_doc:
                    continue
                if example['documents'][i]['is_selected'] == "true":
                    continue
                for para in example['documents'][i]['segmented_paragraphs']:
                    doc_tokens.extend(para)
                break
            assert doc_tokens
            true_example = get_train(question, str(q_id) + "_0", zhidao_dict[q_id]["doc_tokens"], fake_answer, zhidao_dict[q_id]["answer_span"], "false", "zhidao")
            false_example = get_train(question,str(q_id)+"_1",doc_tokens,"","","true","zhidao")
            writer.write(json.dumps(true_example,ensure_ascii=False))
            writer.write('\n')
            writer.write(json.dumps(false_example,ensure_ascii=False))
            writer.write("\n")
            useful_count+=1

    search_dict = {}
    id = []
    with open(lhy_search,"r", encoding='utf-8') as reader:
        for line in reader:
            example = json.loads(line)
            qas_id = example['question_id']
            search_dict[int(qas_id)] = {"question":example['question'],
                                        "doc_tokens":example['doc_tokens'],
                                        "answer_span":example['answer_span'],
                                        "fake_answer":example['fake_answer']
                                        }
    id = list(search_dict.keys())

    with open(source_search,'r', encoding='utf-8') as reader:
        for line in reader:
            all_count +=1
            example = json.loads(line)

            if int(example['question_id']) not in id:
                no_id+=1
                continue
            try:
                assert len(example["documents"]) > 1
            except AssertionError:
                # print(q_id)
                one_doc+=1
                continue
            q_id = int(example['question_id'])
            question = example['question']
            fake_answer = example['fake_answers']
            assert fake_answer == search_dict[q_id]['fake_answer']
            ans_doc = int(example['answer_docs'][0])


            for i in range(len(example['documents'])-1,-1,-1):
                doc_tokens = []
                if i == ans_doc:
                    continue
                if example['documents'][i]['is_selected'] == "true":
                    continue
                if len(example['documents'][i]['segmented_paragraphs'][0])==0:
                    continue
                for para in example['documents'][i]['segmented_paragraphs']:
                    doc_tokens.extend(para)
            if doc_tokens ==[]:
                no_doc+=1
                print(q_id)
                continue
            true_example = get_train(question, str(q_id) + "_0", search_dict[q_id]["doc_tokens"], fake_answer,
                                     search_dict[q_id]["answer_span"], "false", "search")
            false_example = get_train(question,str(q_id)+"_1",doc_tokens,"","","true","search")


            writer.write(json.dumps(true_example,ensure_ascii=False))
            writer.write('\n')
            writer.write(json.dumps(false_example,ensure_ascii=False))
            writer.write("\n")
            useful_count+=1

    print(all_count)
    print(no_id)
    print(one_doc)
    print(no_doc)
    print(useful_count)
    writer.close()


def build_dev_v2(source_zhidao="F:\\baidu_raw_data\dev_preprocessed\dev_preprocessed\devset\zhidao.dev.json",
                 source_search="F:\\baidu_raw_data\dev_preprocessed\dev_preprocessed\devset\search.dev.json",
                 out="F:\Baidu_train\\dev_v2.json"):
    def get_dev(q_id,question,doc_tokens):
        return{
            'question_id':q_id,
            "question":question,
            "doc_tokens":doc_tokens
        }
    writer = open(out,'w',encoding='utf-8')
    with open(source_zhidao,'r', encoding='utf-8') as reader:
        for line in reader:
            example = json.loads(line)

            q_id = int(example['question_id'])
            question = example['question']
            for i in range(len(example['documents'])):
                doc_tokens = []
                for para in example['documents'][i]['segmented_paragraphs']:
                    doc_tokens.extend(para)
                dev_example = get_dev(str(q_id)+'_'+str(i),question,doc_tokens)
                writer.write(json.dumps(dev_example,ensure_ascii=False))
                writer.write('\n')

    with open(source_search,'r', encoding='utf-8') as reader:
        for line in reader:
            example = json.loads(line)

            q_id = int(example['question_id'])
            question = example['question']

            for i in range(len(example['documents'])):
                doc_tokens = []
                for para in example['documents'][i]['segmented_paragraphs']:
                    doc_tokens.extend(para)
                dev_example = get_dev(str(q_id)+'_'+str(i),question,doc_tokens)

                writer.write(json.dumps(dev_example,ensure_ascii=False))
                writer.write('\n')

def filter_top_doc_index(null_odd="C:\\Users\workstation\Desktop\\null_odds.json",
                         output ="C:\\Users\workstation\Desktop\\id_dev_search.json"):
    with open(null_odd,'r',encoding='utf-8') as f:
        data = json.load(f)
    ids = list(data.keys())
    raw_id ={}
    for id in ids:
        if id[:-2] not in raw_id.keys():
            raw_id[id[:-2]] = []

    for id in list(raw_id.keys()):
        odds =[]
        for i in range(5):
            if id+'_'+str(i) in ids:
                odds.append(data[id+'_'+str(i)])
        raw_id[id]=odds.index(max(odds))

    new_ids = list(raw_id.keys())
    writer = open(output,'w',encoding='utf-8')
    writer.write(json.dumps(raw_id,indent=4))
