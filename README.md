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

## 3. 进展提交汇总

| 日期  |      表现       | 提升 |                            改进点                            |
| :---: | :-------------: | :--: | :----------------------------------------------------------: |
| 03/31 | R:37.15 B:23.41 |  --  | 文章选用与问题recall最高的一个文章<br />答案选取模型只跑了3个epoch，未进一步测试 |
| 04/01 | R:40.84 B:24.82 | 3.69 | 每一个文章选用最相关的段落进行拼接<br />答案选取仍然3个epoch。 |
|       |                 |      |                                                              |
|       |                 |      |                                                              |



## 4. 模型结构

我们的模型初步定为三个bert，简称Tri-Bert。

### 模型的流程图
![模型流程图](http://d.hiphotos.baidu.com/image/%70%69%63/item/aec379310a55b319b8172d674da98226cffc1731.jpg){:height="50%" width="50%"}
### Passage Reranking
![passage reranking](http://f.hiphotos.baidu.com/image/%70%69%63/item/96dda144ad34598277664b8002f431adcbef8430.jpg){:height="50%" width="50%"}
### Answer Prediction
![Answer Prediction](http://f.hiphotos.baidu.com/image/%70%69%63/item/0bd162d9f2d3572c6cbe35ce8413632762d0c340.jpg){:height="50%" width="50%"}

## 5. 任务分配

小组目前成员5名：任慕成、魏然、柏宇、王洋、刘宏玉。



