> TriB-QA


###### **富强，民主，文明，和谐，自由，平等，公正，法治，爱国，敬业，诚信，友善**

Here is our group's project for CCF & Baidu 2019 Reading Comprehension Competition, dataset is Baidu Dureader.

The code will be completed once the competition finishes.
The whole project is based on pytorch-version BERT: Passage Rerank, Answer predcition and YES/NO answer classification. So you may need to download pretrained language model,config file and vocab list in advance, or use our pretrained model to get final prediction. 

Later, if possible, we will build a simple pipeline to ease the complicated procedures, also a web API may be built for it.


## 1.任务介绍  Task Description

[百度阅读理解竞赛官网](http://lic2019.ccf.org.cn/read)查看具体要求.

当前训练数据集可以从我这边用U盘拷贝。

项目注意事项、进展情况可以由[这里](https://github.com/trib-plan/TriB-QA/projects/1)跟踪。

数据格式可以从[这里](https://github.com/trib-plan/TriB-QA/tree/master/data) 查看并补全。

## 2. 时间规划 Time Management

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

## 3. 进展提交汇总  Submitted Result History

| 名字                     | Rouge | Bleu  | 时间      | Rouge提升 | Bleu 提升 |
| ------------------------ | ----- | ----- | --------- | --------- | --------- |
| BIT_03/31(simple)        | 37.15 | 23.41 | 2019/3/31 |           |           |
| 冲鸭04/01(single）       | 40.84 | 24.82 | 2019/4/1  | 3.69      | 1.41      |
| 冲鸭^2(single)           | 44.54 | 27.6  | 2019/4/2  | 3.7       | 2.78      |
| 冲鸭^3(single)           | 45.91 | 35    | 2019/4/8  | 1.37      | 7.4       |
| 冲鸭^4(single)           | 47.85 | 45.61 | 2019/4/9  | 1.94      | 10.61     |
| 冲鸭^5(single)           | 48.03 | 46.09 | 2019/4/10 | 0.18      | 0.48      |
| 冲鸭^6(果断就会白给)     | 48.13 | 46.7  | 2019/4/12 | 0.1       | 0.61      |
| 冲鸭^7(single)           | 50.3  | 52.77 | 2019/4/15 | 2.17      | 6.07      |
| 冲鸭^8(single)           | 48.27 | 49.65 | 2019/5/5  | -2.03     | -3.12     |
| 冲鸭^8(single)           | 46.35 | 48.07 | 2019/5/6  | -1.92     | -1.58     |
| 冲鸭^8(single)           | 50.46 | 52.37 | 2019/5/7  | 4.11      | 4.3       |
| 冲鸭^9(single)           | 52.5  | 54.3  | 2019/5/8  | 2.04      | 1.93      |
| 冲鸭^10(single)          | 53.13 | 54.63 | 2019/5/12 | 0.63      | 0.33      |
| 冲鸭^11(single)          | 54.12 | 55.82 | 2019/5/13 | 0.99      | 1.19      |
| 果断就会白给(single)     | 54.54 | 55.87 | 2019/5/15 | 0.42      | 0.05      |
| 果断就会白给(single)     | 54.47 | 55.67 | 2019/5/16 | -0.07     | -0.2      |
| 果断就会白给(single)     | 54.97 | 56.05 | 2019/5/18 | 0.5       | 0.38      |
| 果然还是白给了吗(single) | 55.3  | 56.09 | 2019/5/19 | 0.33      | 0.04      |
| ....                     |       |       |           | 18.15     | 32.68     |



## 4. 模型结构 Model Structure

我们的模型初步定为三个bert，简称Tri-Bert。
Since Our model was initialy desgined as a pipileline include three BERT, we also call it TriB(ert) in our group :>. Sounds rouge but superisely effective.

### 模型的流程图  Model Flow-Chart
<img src ="http://d.hiphotos.baidu.com/image/%70%69%63/item/aec379310a55b319b8172d674da98226cffc1731.jpg" width= "80%" height="60%"/>

### Passage Reranking

<img src ="http://f.hiphotos.baidu.com/image/%70%69%63/item/96dda144ad34598277664b8002f431adcbef8430.jpg" width= "100%" height="60%"/>

### Answer Prediction

<img src ="http://f.hiphotos.baidu.com/image/%70%69%63/item/0bd162d9f2d3572c6cbe35ce8413632762d0c340.jpg" width= "100%" height="100%"/>

## 5. 任务分配 Task Allocation
小组目前成员5名：任慕成、魏然、柏宇、王洋、刘宏玉。



