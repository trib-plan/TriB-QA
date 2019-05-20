import json
# 自己做测试集和训练集。
from preprocess_rougel import get_rougel
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
# 一开始的train数据拆开 做一个训练集和开发集。
def generate_train_dev_from_raw_train():
    cnt = 0
    f_cnt = 0
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

# 最后用所有的train 例子去训练Bert模型，最后是25W个例子，不需要开发集。
def generate_all_train_data():
    f_cnt = 0
    cnt = 0
    filter_answer = 0
    no_true_answer = 0
    low_rougel= 0
    writer = open("F:\Baidu_train\\filtered_all_train.json",'w',encoding='utf-8')
    with open("F:\\baidu_raw_data\\train_preprocessed\\train_preprocessed\\trainset\\search.train.json",encoding = 'utf-8') as reader:
    # with open("F:\TriB-QA\Intermediate_data\search.dev.json",
    #           encoding='utf-8') as reader:
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

            answer_text =selected_doc['segmented_paragraphs'][selected_para_no]
            fake_answer = example['fake_answers']
            answer_spans = example['answer_spans'][0]
            answers = "".join(answer_text[answer_spans[0]: answer_spans[1] + 1])
            if answers in ['。', '.', ',', '，', '!', '！', '?', '？', '的', '了', '%', ]:
                filter_answer += 1
                continue
            elif len(answer_text[answer_spans[0]: answer_spans[1] + 1]) < 20 and example['match_scores'][0] < 0.5:
                filter_answer += 1
                continue
            true_answer = example['answers']
            try:
                rougel = get_rougel(fake_answer[0], true_answer)
            except:
                no_true_answer += 1
                rougel = 1
            if rougel < 0.5:
                low_rougel += 1
                continue

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
            writer.write(json.dumps(example, ensure_ascii=False))
            writer.write('\n')
        print("%d / %d " % (f_cnt,cnt))

    print('finishing search train')

    with open("F:\\baidu_raw_data\\train_preprocessed\\train_preprocessed\\trainset\\zhidao.train.json",encoding = 'utf-8') as reader:
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


            try:
                selected_doc = example['documents'][answer_doc]
            except IndexError:
                print(example)
                continue
            selected_para_no = int(selected_doc['most_related_para'])

            answer_text =selected_doc['segmented_paragraphs'][selected_para_no]
            fake_answer = example['fake_answers']
            answer_spans = example['answer_spans'][0]
            answers = "".join(answer_text[answer_spans[0]: answer_spans[1] + 1])
            if answers in ['。', '.', ',', '，', '!', '！', '?', '？', '的', '了', '%', ]:
                filter_answer += 1
                continue
            elif len(answer_text[answer_spans[0]: answer_spans[1] + 1]) < 20 and example['match_scores'][0] < 0.5:
                filter_answer += 1
                continue
            true_answer = example['answers']
            try:
                rougel = get_rougel(fake_answer[0], true_answer)
            except:
                no_true_answer += 1
                rougel = 1
            if rougel < 0.5:
                low_rougel += 1
                continue


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
            writer.write(json.dumps(example, ensure_ascii=False))
            writer.write('\n')
        print("%d / %d " % (f_cnt,cnt))

    writer.close()


#
# def generate_all_train_data():
#     f_cnt = 0
#     cnt = 0
#     writer = open("F:\TriB-QA\Intermediate_data\\official_dev.json",'w',encoding='utf-8')
#     # with open("D:\迅雷下载\\train_preprocessed\\train_preprocessed\\trainset\\search.train.json",encoding = 'utf-8') as reader:
#     with open("F:\TriB-QA\Intermediate_data\search.dev.json",
#               encoding='utf-8') as reader:
#         for line in reader:
#             selected_segmented_para = []
#             selected_para = []
#             example = json.loads(line)
#             cnt +=1
#             if len(example['answer_spans']) == 0:
#                 continue
#             start_position = int(example['answer_spans'][0][0])
#             end_position = int(example['answer_spans'][0][1])
#
#
#             answer_doc = int(example['answer_docs'][0])
#             selected_doc = example['documents'][answer_doc]
#             selected_para_no = int(selected_doc['most_related_para'])
#
#             if selected_para_no == 0:
#                 start_position = start_position
#                 end_position = end_position
#                 if selected_para_no + 1 < len(selected_doc['paragraphs']):
#                     selected_para = selected_doc['paragraphs'][selected_para_no] + selected_doc['paragraphs'][
#                         selected_para_no + 1]
#                     selected_segmented_para = selected_doc['segmented_paragraphs'][selected_para_no] + selected_doc['segmented_paragraphs'][selected_para_no + 1]
#                 else:
#                     selected_para = selected_doc['paragraphs'][selected_para_no]
#                     selected_segmented_para = selected_doc['segmented_paragraphs'][selected_para_no]
#             else:
#                 start_position = start_position + len(selected_doc['segmented_paragraphs'][selected_para_no-1])
#                 end_position = end_position +len(selected_doc['segmented_paragraphs'][selected_para_no-1])
#                 if selected_para_no + 1 < len(selected_doc['paragraphs']):
#                     selected_para = selected_doc['paragraphs'][selected_para_no - 1] + selected_doc['paragraphs'][
#                         selected_para_no] + selected_doc['paragraphs'][selected_para_no + 1]
#                     selected_segmented_para = selected_doc['segmented_paragraphs'][selected_para_no-1]+selected_doc['segmented_paragraphs'][selected_para_no]+selected_doc['segmented_paragraphs'][selected_para_no+1]
#                 else:
#                     selected_para = selected_doc['paragraphs'][selected_para_no - 1] + selected_doc['paragraphs'][
#                         selected_para_no]
#                     selected_segmented_para = selected_doc['segmented_paragraphs'][selected_para_no - 1] + \
#                                               selected_doc['segmented_paragraphs'][selected_para_no]
#
#             qas_id = example['question_id']
#             fake_answer = example['fake_answers']
#             question = example['question']
#             example = get_example(qas_id,selected_para,selected_segmented_para,fake_answer,question,start_position,end_position)
#
#             f_cnt += 1
#             writer.write(json.dumps(example, ensure_ascii=False))
#             writer.write('\n')
#         print("%d / %d " % (f_cnt,cnt))
#
#     print('finishing search train')
#
#     with open("F:\TriB-QA\Intermediate_data\zhidao.dev.json",encoding = 'utf-8') as reader:
#         for line in reader:
#             selected_segmented_para = []
#             selected_para = []
#             example = json.loads(line)
#             cnt +=1
#             if len(example['answer_spans']) == 0:
#                 continue
#             start_position = int(example['answer_spans'][0][0])
#             end_position = int(example['answer_spans'][0][1])
#
#
#             answer_doc = int(example['answer_docs'][0])
#             try:
#                 selected_doc = example['documents'][answer_doc]
#             except IndexError:
#                 print(example)
#                 continue
#             selected_para_no = int(selected_doc['most_related_para'])
#
#             if selected_para_no == 0:
#                 start_position = start_position
#                 end_position = end_position
#                 if selected_para_no + 1 < len(selected_doc['paragraphs']):
#                     selected_para = selected_doc['paragraphs'][selected_para_no] + selected_doc['paragraphs'][
#                         selected_para_no + 1]
#                     selected_segmented_para = selected_doc['segmented_paragraphs'][selected_para_no] + selected_doc['segmented_paragraphs'][selected_para_no + 1]
#                 else:
#                     selected_para = selected_doc['paragraphs'][selected_para_no]
#                     selected_segmented_para = selected_doc['segmented_paragraphs'][selected_para_no]
#             else:
#                 start_position = start_position + len(selected_doc['segmented_paragraphs'][selected_para_no-1])
#                 end_position = end_position +len(selected_doc['segmented_paragraphs'][selected_para_no-1])
#                 if selected_para_no + 1 < len(selected_doc['paragraphs']):
#                     selected_para = selected_doc['paragraphs'][selected_para_no - 1] + selected_doc['paragraphs'][
#                         selected_para_no] + selected_doc['paragraphs'][selected_para_no + 1]
#                     selected_segmented_para = selected_doc['segmented_paragraphs'][selected_para_no-1]+selected_doc['segmented_paragraphs'][selected_para_no]+selected_doc['segmented_paragraphs'][selected_para_no+1]
#                 else:
#                     selected_para = selected_doc['paragraphs'][selected_para_no - 1] + selected_doc['paragraphs'][
#                         selected_para_no]
#                     selected_segmented_para = selected_doc['segmented_paragraphs'][selected_para_no - 1] + \
#                                               selected_doc['segmented_paragraphs'][selected_para_no]
#
#             qas_id = example['question_id']
#             fake_answer = example['fake_answers']
#             question = example['question']
#             example = get_example(qas_id,selected_para,selected_segmented_para,fake_answer,question,start_position,end_position)
#
#             f_cnt += 1
#             writer.write(json.dumps(example, ensure_ascii=False))
#             writer.write('\n')
#         print("%d / %d " % (f_cnt,cnt))
#
#     writer.close()

#
# def generate_all_train_data():
#     f_cnt = 0
#     cnt = 0
#     writer = open("F:\TriB-QA\Intermediate_data\\search_dev_paragraph.json", 'w', encoding='utf-8')
#     # with open("D:\迅雷下载\\train_preprocessed\\train_preprocessed\\trainset\\search.train.json",encoding = 'utf-8') as reader:
#     with open("F:\TriB-QA\Intermediate_data\search.dev.json",
#               encoding='utf-8') as reader:
#         for line in reader:
#             example = json.loads(line)
#             cnt += 1
#             selected_doc = example['documents'][0]
#             selected_para_no = int(selected_doc['most_related_para'])
#
#             if selected_para_no == 0:
#                 if selected_para_no + 1 < len(selected_doc['paragraphs']):
#                     selected_segmented_para = selected_doc['segmented_paragraphs'][selected_para_no] + \
#                                               selected_doc['segmented_paragraphs'][selected_para_no + 1]
#                 else:
#                     selected_segmented_para = selected_doc['segmented_paragraphs'][selected_para_no]
#             else:
#                 if selected_para_no + 1 < len(selected_doc['paragraphs']):
#                     selected_segmented_para = selected_doc['segmented_paragraphs'][selected_para_no - 1] + \
#                                               selected_doc['segmented_paragraphs'][selected_para_no] + \
#                                               selected_doc['segmented_paragraphs'][selected_para_no + 1]
#                 else:
#                     selected_segmented_para = selected_doc['segmented_paragraphs'][selected_para_no - 1] + \
#                                               selected_doc['segmented_paragraphs'][selected_para_no]
#
#             qas_id = example['question_id']
#             question = example['question']
#             example = get_example(qas_id, selected_segmented_para, question)
#
#             f_cnt += 1
#             writer.write(json.dumps(example, ensure_ascii=False))
#             writer.write('\n')
#         print("%d / %d " % (f_cnt, cnt))
#
#     print('finishing search train')
#
#     writer1 = open("F:\TriB-QA\Intermediate_data\\zhidao_dev_paragraph.json", 'w', encoding='utf-8')
#     with open("F:\TriB-QA\Intermediate_data\zhidao.dev.json", encoding='utf-8') as reader:
#         for line in reader:
#             example = json.loads(line)
#             cnt += 1
#
#             selected_doc = example['documents'][0]
#             selected_para_no = int(selected_doc['most_related_para'])
#
#             if selected_para_no == 0:
#                 if selected_para_no + 1 < len(selected_doc['paragraphs']):
#                     selected_segmented_para = selected_doc['segmented_paragraphs'][selected_para_no] + \
#                                               selected_doc['segmented_paragraphs'][selected_para_no + 1]
#                 else:
#                     selected_segmented_para = selected_doc['segmented_paragraphs'][selected_para_no]
#             else:
#                 if selected_para_no + 1 < len(selected_doc['paragraphs']):
#                     selected_segmented_para = selected_doc['segmented_paragraphs'][selected_para_no - 1] + \
#                                               selected_doc['segmented_paragraphs'][selected_para_no] + \
#                                               selected_doc['segmented_paragraphs'][selected_para_no + 1]
#                 else:
#                     selected_segmented_para = selected_doc['segmented_paragraphs'][selected_para_no - 1] + \
#                                               selected_doc['segmented_paragraphs'][selected_para_no]
#
#             qas_id = example['question_id']
#             question = example['question']
#             example = get_example(qas_id, selected_segmented_para, question)
#
#             f_cnt += 1
#             writer1.write(json.dumps(example, ensure_ascii=False))
#             writer1.write('\n')
#         print("%d / %d " % (f_cnt, cnt))
#     writer1.close()
#     writer.close()
