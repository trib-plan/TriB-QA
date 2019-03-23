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

### Yes or No classifying task(柏宇 & 王洋)

#### 任务目标

将Yes_No类型的问题答案进行分类，分类结果包括Yes, No以及Depends

#### 任务方案

将拟备选句子输入BERT模型。并将句子的CLS输入一个分类器（前馈网络），经由Softmax层进行结果预测。

#### 任务规划

- [x] 对训练数据集进行处理，得到所有YES_NO类型的answers(不是fake_answers)以及其对应的yesno_answers
- [x] 阅读模型代码，对所得到的数据进行处理以适应模型要求。
- [x] 修改模型代码
- [x] 训练模型并得到初步结果

#### 文档记录
##### 训练级初步处理
###### 操作步骤
1. 将训练数据源文件放入data文件夹下
2. 使用data_filter.py(将程序中的路径改为对应文件名，这里应该写一个parser来在运行时把文件名传进去)对原始数据进行处理，得到finished.json文件(这里在写完parser之后改为finished+文件名.json)。
3. finished.json中的key为answer原句，value为另外两个key-value对，其中包括原句的分词结果"segmented_answers"以及原句的判断结果"yesno_answers"
###### 样例
```
{
    "ipad越狱与未越狱都是可以安装360的": 
       {
            "yesno_answers": "Yes", 
            "segmented_answers": ["ipad", "越狱", "与", "未", "越狱", "都是", "可以", "安装", "360", "的"]
       }
}
```

##### 目前试验结果
在batch_size为32，训练两个epoch的时候，测试集表现最好，准确率达到70.5%

现在在尝试改变训练集和测试集的比例。




