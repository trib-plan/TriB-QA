import json
# A = []
cnt = 0
f_cnt = 0
import codecs
import thulac

def get_example(qas_id, para, seg_para, fake_answer,question,start_poi, end_poi):
    return {
        "id": qas_id,
        "doc": para,
        "seg_para": seg_para,
        "fake_answer": fake_answer,
        "question": question,
        # "seg_qus": seg_qus,
        "answer_span": [start_poi,end_poi]
        }

writer = open("F:\\TriB-QA\data\\f2.search.train.json",'w',encoding='utf-8')
dev_writer = open("F:\\TriB-QA\data\\f2.search.dev.json",'w',encoding='utf-8')
with open("D:\迅雷下载\\train_preprocessed\\train_preprocessed\\trainset\\search.train.json",encoding = 'utf-8') as reader:
    for line in reader:
        selected_segmented_para = []
        selected_para = []
        example = json.loads(line)
        cnt +=1
        if len(example['answer_spans']) == 0:
            continue
        start_position = int(example['answer_spans'][0][0])
        end_position = int(example['answer_spans'][0][1])


        answer_doc = int(example['answer_docs'][0])
        selected_doc = example['documents'][answer_doc]
        selected_para_no = int(selected_doc['most_related_para'])

        if selected_para_no == 0:
            start_position = start_position
            end_position = end_position
            if selected_para_no + 1 < len(selected_doc['paragraphs']):
                selected_para = selected_doc['paragraphs'][selected_para_no] + selected_doc['paragraphs'][
                    selected_para_no + 1]
                selected_segmented_para = selected_doc['segmented_paragraphs'][selected_para_no] + selected_doc['segmented_paragraphs'][selected_para_no + 1]
            else:
                selected_para = selected_doc['paragraphs'][selected_para_no]
                selected_segmented_para = selected_doc['segmented_paragraphs'][selected_para_no]
        else:
            start_position = start_position + len(selected_doc['segmented_paragraphs'][selected_para_no-1])
            end_position = end_position +len(selected_doc['segmented_paragraphs'][selected_para_no-1])
            if selected_para_no + 1 < len(selected_doc['paragraphs']):
                selected_para = selected_doc['paragraphs'][selected_para_no - 1] + selected_doc['paragraphs'][
                    selected_para_no] + selected_doc['paragraphs'][selected_para_no + 1]
                selected_segmented_para = selected_doc['segmented_paragraphs'][selected_para_no-1]+selected_doc['segmented_paragraphs'][selected_para_no]+selected_doc['segmented_paragraphs'][selected_para_no+1]
            else:
                selected_para = selected_doc['paragraphs'][selected_para_no - 1] + selected_doc['paragraphs'][
                    selected_para_no]
                selected_segmented_para = selected_doc['segmented_paragraphs'][selected_para_no - 1] + \
                                          selected_doc['segmented_paragraphs'][selected_para_no]

        qas_id = example['question_id']
        fake_answer = example['fake_answers']
        question = example['question']
        example = get_example(qas_id,selected_para,selected_segmented_para,fake_answer,question,start_position,end_position)

        f_cnt += 1
        if f_cnt % 15 == 0:
            dev_writer.write(json.dumps(example, ensure_ascii=False))
            dev_writer.write('\n')
        else:
            writer.write(json.dumps(example, ensure_ascii=False))
            writer.write('\n')
    print("%d / %d " % (f_cnt,cnt))


