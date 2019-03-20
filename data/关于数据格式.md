# 关于数据格式 

## Train_Processed

| Example        |                           |                      |                                                              |
| -------------- | ------------------------- | -------------------- | ------------------------------------------------------------ |
|                | Example['Document'] [0-4] | is_selected: True    | True: Answer 从此doc挑选, 反之                               |
|                |                           | title                | 文章的标题                                                   |
|                |                           | most_related_para    | is_selected为True的话，代表正确答案最相关的那一段。 如果为Flase，则为-1 |
|                |                           | segmented_title      | 分词后的title，可进一步被bert的分词拆解                      |
|                |                           | segmented_paragraphs | 分词后的段落                                                 |
|                |                           | paragraphs           | [[p1],[p2],[p3]...]这样格式保留，一个doc可能拥有很多p(aragraphs) |
|                | answer_spans              |                      | fake answer 答案的范围，有的没有                             |
|                | fake_answer               |                      | 官方给出的，只有一个                                         |
|                | question                  |                      | 问题                                                         |
|                | segmented_answers         |                      | 答案的拆分                                                   |
|                | answers                   |                      | 真正的答案，数量根据前面is_selected 判断                     |
|                | answer_docs               |                      | 应该是fake answer 所在的文章编号                             |
|                | segmented_question        |                      | 拆分的问题                                                   |
|                | match_scores              |                      | fake answer 与answer 的分数比较                              |
|                | fact_or_opinion           |                      | OPINION，FACT                                                |
|                | question_id               |                      | 问题编号，必须有                                             |
| 特殊情况的处理 |                           |                      |                                                              |
|                | question_type             | DESCRIPTION          | 这类问题没有额外的信息                                       |
|                |                           | ENTITY               | entity_answers:[[e1] [e2]...] 从每个answer 抽出一个(?)信息。 |
|                |                           | YES_NO               | yesno_type :OPINION，FACT<br />yesno_answer: [ 'YES'/ 'NO'/'DEPENDS'] 根据answer 的个数判断数量。 |

