> TriB-QA

###### **富强，民主，文明，和谐，自由，平等，公正，法治，爱国，敬业，诚信，友善**
## 1.任务介绍

[百度阅读理解竞赛官网](http://lic2019.ccf.org.cn/read)查看具体要求.  

当前训练数据集可以从我这边用U盘拷贝。

项目注意事项、进展情况可以由[这里](https://github.com/trib-plan/TriB-QA/projects/1)跟踪。

## 2. 时间规划

竞赛的关键时间如下图：
    <table>
        <tr>
            <th>Event</th>
            <th>时间</th>
        </tr>
        <tr>
            <th><font color=red>报名</font> 训练数据发放</th>
            <th>02/25</th>
        </tr>
        <tr>
            <th><font color=red>报名截止</font> 开发数据、测试集1发放</th>
            <th>03/31</th>
        </tr>
        <tr>
            <th>测试集2发放</th>
            <th>05/13</th>
        </tr>
        <tr>
            <th><font color=red>结果提交截止</font></th>
            <th>05/20</th>
        </tr>
        <tr>
            <th>公布结果，接受报告论文
            <th>05/31</th>
        </tr>
    </table>

另外的个人分配时间周四定下来。

## 3. 模型结构

我们的模型初步定为三个bert，简称Tri-Bert。

### 模型的流程图
![模型流程图](http://d.hiphotos.baidu.com/image/%70%69%63/item/aec379310a55b319b8172d674da98226cffc1731.jpg)
### Passage Reranking
![passage reranking](http://f.hiphotos.baidu.com/image/%70%69%63/item/96dda144ad34598277664b8002f431adcbef8430.jpg)
### Answer Prediction
![Answer Prediction](http://f.hiphotos.baidu.com/image/%70%69%63/item/0bd162d9f2d3572c6cbe35ce8413632762d0c340.jpg)

## 4. 任务分配

小组目前成员4名：任慕成、魏然、柏宇、王洋。

任务：paragraph选择（rwei）
在实际数据中，每一个question（String类型）对应一个documents（List类型）
每一个documents，包含多个 paragraph （最少1个，对多5个）
每个 paragraph 具有 title - paragraph - is_selected 三项内容
title：该paragraph的标题，和question文本类似
paragraph：该paragraph的实际文本类容，长度不确定，可能是一句话或者是多句话
is_selected: 该paragraph是否被选中作答案抽取的标签 True or False

在这里抛弃了 title，因为从 search_sampling.txt 和 zhidao_sampling.txt 文件中可以看到title对该paragraph是否被选中关系不大。
title 应该反应了数据集的QA过程：
1. 给定question
2. 搜索和question最相似的title，得到多个候选问题
3. 逐一分辨候选问题，得到最有可能包含答案的paragraph
4. 从paragraph中得到实际answer

在paragraph选择任务中，将文本处理为如下格式
question /t paragraph /t label /n
如果一个document含有多个paragraph，则生成对应个数的上述 数据-标签 对，并按照 80% - 20%分为训练集和测试集
统计数据如下：
Baidu Search
- baidu_search example num:  632553
- baidu_search avg_question_len:  9.6154535667367
- baidu_search avg_paragraph_len:  774.4874216073594
- baidu_search 1 label num:  238427
- baidu_search 0 label num:  394126

Baidu Zhidao
- baidu_zhidao example num:  653065
- baidu_zhidao avg_question_len:  9.523424161454066
- baidu_zhidao avg_paragraph_len:  591.300689824137
- baidu_zhidao 1 label num:  351186
- baidu_zhidao 0 label num:  301879



