## 第一步 分别预测search 和zhidao，同时跑快一点

需要自行定义predict file，output file （输出放在自己名字底下好区分）

```
export BAIDU_DIR=/home/share/Baidu_Dev/
CUDA_VISIBLE_DEVICES=4  python run_Everything.py \
  --bert_model /home/share/Baidu_Dev/raw_data\
  --do_predict \
  --do_lower_case \
  --predict_file $BAIDU_DIR/rmc/zhidao_dev_paragraph.json\
  --train_batch_size 10 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 400 \
  --doc_stride 400 \
  --max_answer_length 150 \
  --output_dir /home/share/Baidu_Dev/rmc/zhidao/
```

```
export BAIDU_DIR=/home/share/Baidu_Dev/
CUDA_VISIBLE_DEVICES=5  python run_Everything.py \
  --bert_model /home/share/Baidu_Dev/raw_data\
  --do_predict \
  --do_lower_case \
  --predict_file $BAIDU_DIR/rmc/search_dev_paragraph.json\
  --train_batch_size 10 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 400 \
  --doc_stride 400 \
  --max_answer_length 150 \
  --output_dir /home/share/Baidu_Dev/rmc/search/
```

## 第二步  判断YES 并补齐原本答案

data则是raw_data地址，bert model 为训练好的模型所在地，**无需改变**

 **需要改变**：

zhidao search file 是上一步你分别输出预测的output地址。
output 是下一个阶段的输入

```
CUDA_VISIBLE_DEVICES=5 python run_classifier.py \
--data_dir=/home/share/Baidu_Dev/raw_data \
--bert_model=/home/share/Baidu_Dev/yesno_data \
--task_name=baidu \
--do_eval \
--do_lower_case \
--zhidao_file=/home/share/Baidu_Dev/rmc/zhidao \
--search_file=/home/share/Baidu_Dev/rmc/search \
--output_dir=/home/share/Baidu_Dev/rmc/trial_output \
```



## 第三步 进入到evaluation metric 文件夹 输入以下命令

第一个为你上阶段的输出的文件位置 **需要改变**
第二个为固定的refjson 已经在raw data里面**无需改变**
最后v1别扔了

```
python mrc_eval.py /home/share/Baidu_Dev/rmc/trial_output/pred.json /home/share/Baidu_Dev/raw_data/ref.json v1
```

