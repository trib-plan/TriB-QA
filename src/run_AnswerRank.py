import logging
import os
import argparse
import random
from tqdm import tqdm, trange
import json
import re

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from Bert_tokenization import BertTokenizer
from Bert_modeling import BertForMultipleChoice
from Bert_optimization import BertAdam
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class Baidu_dev_example(object):
    """A single training/test example for the SWAG dataset."""
    def __init__(self,
                 qas_id,
                 type,
                 question,
                 choice_0,
                 choice_1,
                 choice_2,
                 choice_3,
                 choice_4,
                 choice_5,
                 choice_6,
                 choice_7,
                 choice_8,
                 choice_9,
                 choice_10,
                 choice_11,
                 choice_12,
                 choice_13,
                 choice_14,
                 choice_15,
                 choice_16,
                 choice_17,
                 choice_18,
                 choice_19,
                 label = None):
        self.qas_id = qas_id
        self.type = type
        self.question = question
        self.choices = [
            choice_0,
            choice_1,
            choice_2,
            choice_3,
            choice_4,
            choice_5,
            choice_6,
            choice_7,
            choice_8,
            choice_9,
            choice_10,
            choice_11,
            choice_12,
            choice_13,
            choice_14,
            choice_15,
            choice_16,
            choice_17,
            choice_18,
            choice_19,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"qas_id: {self.qas_id}",
            f"type: {self.type}",
            f"question: {self.question}",
            f"choice_0: {self.choices[0]}",
            f"choice_1: {self.choices[1]}",
            f"choice_2:{self.choices[2]}",
            f"choice_3:{self.choices[3]}",
            f"choice_4:{self.choices[4]}",
            f"choice_5:{self.choices[5]}",
            f"choice_6:{self.choices[6]}",
            f"choice_7:{self.choices[7]}",
            f"choice_8:{self.choices[8]}",
            f"choice_9:{self.choices[9]}",
            f"choice_10:{self.choices[10]}",
            f"choice_11:{self.choices[11]}",
            f"choice_12:{self.choices[12]}",
            f"choice_13:{self.choices[13]}",
            f"choice_14:{self.choices[14]}",
            f"choice_15:{self.choices[15]}",
            f"choice_16:{self.choices[16]}",
            f"choice_17:{self.choices[17]}",
            f"choice_18:{self.choices[18]}",
            f"choice_19:{self.choices[19]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label

def clean_space(text):
    """"
    处理多余的空格
    """
    match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
    should_replace_list = match_regex.findall(text)
    order_replace_list = sorted(should_replace_list,key=lambda i:len(i),reverse=True)
    for i in order_replace_list:
        if i == u' ':
            continue
        new_i = i.strip()
        text = text.replace(i,new_i)
    return text

def read_baidu_dev_example(input_file,is_training=True):
    with open(input_file, "r", encoding='utf-8') as reader:
        examples = []
        for line in reader:
            example = json.loads(line)
            uni_id = example['question_id']
            question = example['question']
            type = example['question_type']
            answers = example['answers']
            choices = []
            rouge = []
            for answer in answers:
                choices.append(clean_space(answer['text']))
                rouge.append(answer['RougeL'])

            label = rouge.index(max(rouge))

            example = Baidu_dev_example(
                qas_id=uni_id,
                type=type,
                question=question,
                choice_0=choices[0],
                choice_1=choices[1],
                choice_2=choices[2],
                choice_3=choices[3],
                choice_4=choices[4],
                choice_5=choices[5],
                choice_6=choices[6],
                choice_7=choices[7],
                choice_8=choices[8],
                choice_9=choices[9],
                choice_10=choices[10],
                choice_11=choices[11],
                choice_12=choices[12],
                choice_13=choices[13],
                choice_14=choices[14],
                choice_15=choices[15],
                choice_16=choices[16],
                choice_17=choices[17],
                choice_18=choices[18],
                choice_19=choices[19],
                label=label
            )
            examples.append(example)
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # MCScript is a multiple choice task.
    # Each choice will correspond to a sample on which we run the
    # inference. For a given MCscript example, we will create the 23.
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]

    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 2
    # outputs.
    features = []
    for example_index, example in enumerate(examples):
        context_tokens = tokenizer.tokenize(example.type)
        question_tokens = tokenizer.tokenize(example.question)

        choices_features = []
        for choice_index, choice in enumerate(example.choices):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:] + question_tokens[:]
            ending_tokens = tokenizer.tokenize(choice)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        # if example_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info(f"qas_id: {example.qas_id}")
        #     for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
        #         logger.info(f"choice: {choice_idx}")
        #         logger.info(f"tokens: {' '.join(tokens)}")
        #         logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
        #         logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
        #         logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
        #     if is_training:
        #         logger.info(f"label: {label}")

        features.append(
            InputFeatures(
                example_id = example.qas_id,
                choices_features = choices_features,
                label = label
            )
        )

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def model_eval(model,device,input_examples,do_train=False):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for input_ids, input_mask, segment_ids, label_ids in input_examples:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    if do_train:
        result = {'train_loss': eval_loss,
                  'train_accuracy': eval_accuracy}
        logger.info("***** Train results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    else:
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,}
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

    return eval_accuracy,result

def model_save(model,output_dir,name):
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(output_dir, name)
    torch.save(model_to_save.state_dict(), output_model_file)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .json files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size // args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    # path = "C:\\Users\workstation\PycharmProjects\TransQA\data"
    # tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=True)


    train_examples = None
    num_train_steps = None

    if args.do_train:
        train_examples = read_baidu_dev_example(os.path.join(args.data_dir, 'dev_answer_rerank_train.json'), is_training = True)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = BertForMultipleChoice.from_pretrained(args.bert_model,
        cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
        num_choices=20)


    if args.fp16:
        model.half()
    model.to(device)


    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
    global_step = 0
    if args.do_train:
        train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length, True)
        logger.info("*********Reading Training Examples*********")
        logger.info("   NO. of  Examples: %d", len(train_examples))
        logger.info("   Batch Size: %d", args.train_batch_size)
        logger.info("   NO. steps: %d", num_train_steps)
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        # ***************************************************************************************************
        eval_examples = read_baidu_dev_example(os.path.join(args.data_dir, 'dev_answer_rerank_test.json'),
                                               is_training=True)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length, True)
        logger.info("*********Reading Evaluation Examples*********")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        # ****************************************************************************************************
        best_acc = 0.0
        writer = open(os.path.join(args.output_dir,"eval_results.log"),'w',encoding='utf-8')
        model.train()
        for _ in trange(int(args.num_train_epochs),desc='Epoch'):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0 ,0
            if _ == 0:
                eval_accuracy,re = model_eval(model,device,eval_dataloader)
            for step, batch in enumerate(tqdm(train_dataloader,desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                # rescale loss for fp16 training
                # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                if nb_tr_steps % 20 == 0:
                    print('Iter: %d, Loss: %f, Tr_Loss: %f' % (nb_tr_steps, loss,tr_loss))

            logger.info(f"Epoch: {_+1}")
            # train_accuracy = model_eval(model,device,train_dataloader[:2000],do_train=True)
            eval_accuracy,eval_results= model_eval(model, device, eval_dataloader, do_train=False)
            if eval_accuracy > best_acc:
                best_acc = eval_accuracy
                model_save(model, args.output_dir, name="best_pytorch_model.bin")
            for key in sorted(eval_results.keys()):
                writer.write("%s = %s \n" % (key, str(eval_results[key])))
        model_save(model, args.output_dir, name="pytorch_model.bin")
        writer.close()

    if args.do_eval and not args.do_train:
        output_model_file = os.path.join(args.output_dir,"best_pytorch_model.bin")
        output_eval_file = os.path.join(args.output_dir, "best_eval.log")
        model_state_dict = torch.load(output_model_file)
        model = BertForMultipleChoice.from_pretrained(args.bert_model,
                                                      state_dict=model_state_dict,
                                                      num_choices=20)
        model.to(device)

        eval_examples = read_baidu_dev_example(os.path.join(args.data_dir, 'test-data.json'),
                                               is_training=True)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length, True)
        logger.info("*********Reading Evaluation Examples*********")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        eval_accuracy, eval_results = model_eval(model,device,eval_dataloader,do_train= False)
        with open(output_eval_file,'w') as writer:
            for key in sorted(eval_results.keys()):
                writer.write("%s = %s\n" % (key, str(eval_results[key])))
if __name__ == "__main__":
    main()