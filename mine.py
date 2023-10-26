import torch
from datasets import load_metric
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, BertTokenizerFast, \
    BertForTokenClassification, TrainingArguments, Trainer
from transformers import get_scheduler
import numpy as np
import re
from tqdm.auto import tqdm
import evaluate

#加载数据
def load_conll_file(file_path):
    sentences = []
    labels = []
    pua_pattern = re.compile("[\uE000-\uF8FF]|[\u200b\u200d\u200e]")
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        label = []
        for line in f:
            line = line.strip()#去除换行符
            if len(line) == 0:
                if len(sentence) > 0:
                    sentences.append(sentence)
                    labels.append(label)
                sentence = []
                label = []
            else:
                parts = line.split()
                word = parts[0]
                tag = parts[1]
                word = re.sub(pua_pattern, "", word) #删除这些私有域字符
                if word:
                    sentence.append(word)
                    label.append(tag)
        if len(sentence) > 0:
            sentences.append(sentence)
            labels.append(label)
    return sentences,labels
#['浙', '江', '杭', '州', '市', '江', '干', '区', '九', '堡', '镇', '三', '村', '村', '一', '区'],
#['B-prov', 'E-prov', 'B-city', 'I-city', 'E-city', 'B-district', 'I-district', 'E-district', 'B-town', 'I-town', 'E-town', 'B-community', 'I-community', 'E-community', 'B-poi', 'E-poi']

#加载测试数据
def load_test_file(file_path):
    sentences = []
    labels = []
    pua_pattern = re.compile("[\uE000-\uF8FF]|[\u200b\u200d\u200e]")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            ids, words = line.strip().split('\001')
            # 要预测的数据集没有label，伪造个O，
            words = re.sub(pua_pattern, '', words)
            label=['O' for x in range(0,len(words))]
            sentence=[]
            for c in words:
                sentence.append(c)
            sentences.append(sentence)
            labels.append(label)
    return sentences,labels

train_sentences, train_labels = load_conll_file('./data/train.conll')
train_sentences = train_sentences[:100]
train_labels = train_labels[:100]
dev_sentences,dev_labels = load_conll_file('./data/dev.conll')
dev_sentences[:20]
dev_labels[:20]
# print(train_sentences[:1],train_labels[:1])
# print(dev_sentences[:1],dev_labels[:1])
#[['浙', '江', '杭', '州', '市', '江', '干', '区', '九', '堡', '镇', '三', '村', '村', '一', '区']] [['B-prov', 'E-prov', 'B-city', 'I-city', 'E-city', 'B-district', 'I-district', 'E-district', 'B-town', 'I-town', 'E-town', 'B-community', 'I-community', 'E-community', 'B-poi', 'E-poi']]
#[['杭', '州', '五', '洲', '国', '际']] [['B-city', 'E-city', 'B-poi', 'I-poi', 'I-poi', 'E-poi']]

# 建立tag到id的映射表
tags_list = ['O']
for labels in (train_labels + dev_labels):
    for tag in labels:
        if tag not in tags_list:
            tags_list.append(tag)

tag2id = {tag: i for i, tag in enumerate(tags_list)}
id2tag = {i: tag for i, tag in enumerate(tags_list)}

#数据预处理，转化成Bert模型接受的格式
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
#定义数据集类型
class MyDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, tag2id):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.tag2id = tag2id

        self.encodings = tokenizer(sentences, is_split_into_words=True, padding=True)

        self.encoded_labels = []
        for label, input_id in zip(labels, self.encodings['input_ids']):
            # create an empty array of 0
            t = len(input_id) - len(label) - 1
            label = ['O'] + label + ['O'] * t
            self.encoded_labels.append([tag2id[l] for l in label])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        input_ids = torch.LongTensor(self.encodings['input_ids'][idx])
        attention_mask = torch.LongTensor(self.encodings['attention_mask'][idx])
        labels = torch.LongTensor(self.encoded_labels[idx])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

train_dataset = MyDataset(train_sentences, train_labels, tokenizer, tag2id)#训练集
eval_dataset = MyDataset(dev_sentences, dev_labels, tokenizer, tag2id)#测试集

#定义模型
model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(tags_list))
metric = load_metric("seqeval")

# 定义训练参数TrainingArguments和Trainer
args = TrainingArguments(
    "chi",                     # 输出路径，存放检查点和其他输出文件
    evaluation_strategy="epoch",        # 定义每轮结束后进行评价
    learning_rate=2e-5,                 # 定义初始学习率
    per_device_train_batch_size=16,     # 定义训练批次大小
    per_device_eval_batch_size=16,      # 定义测试批次大小
    num_train_epochs=3,                 # 定义训练轮数
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# 开始训练！（主流GPU上耗时约几分钟）
trainer.train()
trainer.evaluate()

test_sentences, test_labels = load_test_file('./data/final_test.txt')
test_dataset = MyDataset(test_sentences, test_labels, tokenizer, tag2id)
test_dataloader = DataLoader(test_dataset, batch_size=64)

# 指定文件名
file_name = "./data/output.txt"

# 打开文件，以写入模式写入数据
with open(file_name, "w", encoding="utf-8") as file:
    i = 1
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        for input_id, prediction in zip(batch['input_ids'], predictions):
            index = input_id.tolist().index(102)
            sentence = tokenizer.decode(input_id[1:index]).replace(" ", "")
            prediction2 = [id2tag[t.item()] for t in prediction[1:index]]
            prediction_str = ' '.join(prediction2)

            line = f"{i}\u0001{sentence}\u0001{prediction_str}\n"
            file.write(line)
            i += 1