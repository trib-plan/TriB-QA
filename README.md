> TriB-QA

###### **富强，民主，文明，和谐，自由，平等，公正，法治，爱国，敬业，诚信，友善**

## 任务：paragraph选择（rwei）
###### **自己的branch就把做的放在前面了**

### 数据整体统计情况：  
baidu_search  
example_nums: 136208  
doc_nums: 632553  
paragraph_nums: 7371191  
avg_para_charlen: 63.99444784431715  
avg_para_wordlen: 36.97641317393621  
avg_mostrelated_charlen: 209.88829709484418  
avg_mostrelated_wordlen: 120.48022854965568  
most nums: 608883  
  
baidu_zhidao  
example_nums: 135366  
doc_nums: 653065  
paragraph_nums: 1772768  
avg_para_charlen: 215.0283455026264  
avg_para_wordlen: 128.91797855105688  
avg_mostrelated_charlen: 359.0590676297614  
avg_mostrelated_wordlen: 215.21554889436135  
most nums: 631852  
  

### 关于如何从paragraphs当中选取most_related_paragraph:  
1. 他们在测试的时候选取了question和paragraphs中rouge f1值最高的paragraph, 但是从下面的统计数据可以看出来， 选取length最长的paragraph是most_related的正确率要比rouge最高的正确率要高，而且高不少。。。这是因为第一步尽量选择包含信息多的paragraph？？？  
2. 略略略  

  
数据格式：  
指标（length, rouge, length*rouge)：符合条件数-例子总数-比例  
第一行符合条件，指的是most_related_paragraph具有max_length或者max_rouge或者最大两者乘积  
第二行符合条件，指的是most_related_paragraph的指标大于备选paragraphs的平均值  

baidu_search:  
sentence: 100  
lengths:  159 282 0.5638297872340425  
lengths:  231 282 0.8191489361702128  
rouge:  39 282 0.13829787234042554  
rouge:  98 282 0.3475177304964539  
dot:  136 282 0.48226950354609927  
dot:  222 282 0.7872340425531915  
time:  3.0687928199768066  
  
sentence: 1000  
lengths:  1659 2896 0.5728591160220995  
lengths:  2406 2896 0.8308011049723757  
rouge:  418 2896 0.14433701657458564  
rouge:  949 2896 0.32769337016574585  
dot:  1223 2896 0.42230662983425415  
dot:  2285 2896 0.7890193370165746  
time:  28.952566623687744  
  
sentence: 10000  
lengths:  16390 28504 0.5750070165590794  
lengths:  23794 28504 0.8347600336794836  
rouge:  4045 28504 0.14190990738142015  
rouge:  8912 28504 0.3126578725792871  
dot:  12618 28504 0.44267471232107775  
dot:  22566 28504 0.791678360931799  
time:  289.41554856300354  

baidu_zhidao:  
sentence: 100  
lengths:  108 173 0.6242774566473989  
lengths:  133 173 0.7687861271676301  
rouge:  46 173 0.2658959537572254  
rouge:  62 173 0.3583815028901734  
dot:  109 173 0.630057803468208  
dot:  141 173 0.815028901734104  
time:  1.3977971076965332  
  
sentence: 1000  
lengths:  1035 1592 0.6501256281407035  
lengths:  1224 1592 0.7688442211055276  
rouge:  438 1592 0.2751256281407035  
rouge:  570 1592 0.35804020100502515  
dot:  979 1592 0.6149497487437185  
dot:  1274 1592 0.800251256281407  
time:  13.759829044342041  
  
sentence: 10000  
lengths:  10526 16053 0.6557029838659441  
lengths:  12408 16053 0.7729396374509437  
rouge:  4193 16053 0.26119728399676073  
rouge:  5474 16053 0.3409954525633838  
dot:  10154 16053 0.6325297452189622  
dot:  12760 16053 0.794867003052389  
time:  151.27087998390198  
  
### 哇，下面这部分分类看错数据了，真蠢
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

由于 paragraph 长度都很长，实际操作时候取 max_question_len = 400，取 batch_size = 10 ，实测显存占用11g，105服务器单卡最大batch应该在12左右，batch=16实测爆显存，数据处理大概12h一个epoch，目前仍在训练过程当中。
  
## 下面是正常的Tri-Bit  
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




