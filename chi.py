from transformers import get_scheduler
import numpy as np
import re
from tqdm.auto import tqdm
import evaluate
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, BertTokenizerFast
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
    #私用区域字符和文本排版和格式相关，例如零宽度空格、零宽度连接符和左至右标记。
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
            ids, words = line.strip().split('\001')#去除换行，以\001分割
            # 要预测的数据集没有label，伪造个O与word长度相等
            words = re.sub(pua_pattern, '', words)#替换函数
            label=['O' for x in range(0,len(words))]
            sentence=[]
            for c in words:
                sentence.append(c)
            sentences.append(sentence)
            labels.append(label)
    return sentences,labels

train_sentences, train_labels = load_conll_file('./data/train.conll')
print("获取到",len(train_sentences),"条训练数据,",len(train_labels),"个标签\n")
dev_sentences,dev_labels = load_conll_file('./data/dev.conll')
print("获取到",len(dev_sentences),"条测试数据,",len(dev_labels),"个标签")

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
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
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

#以便从数据集中获取单个样本和标签。在这个方法中，你需要实现将原始数据加载、预处理和组织成模型所需的数据结构，比如这里的转化成张量
    def __getitem__(self, idx):#
        input_ids = torch.LongTensor(self.encodings['input_ids'][idx])
        attention_mask = torch.LongTensor(self.encodings['attention_mask'][idx])
        labels = torch.LongTensor(self.encoded_labels[idx])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


train_dataset = MyDataset(train_sentences, train_labels, tokenizer, tag2id)#训练集
eval_dataset = MyDataset(dev_sentences, dev_labels, tokenizer, tag2id)#测试集


#定义模型
model = AutoModelForTokenClassification.from_pretrained('bert-base-chinese', num_labels=len(tag2id))
# 定义Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=64)
print("训练数据封装成批次大小为",train_dataloader.batch_size,"的批次，共",len(train_dataloader),"步")

# 定义优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
print("设定训练",num_epochs,"个周期")
num_training_steps = num_epochs * len(train_dataloader)
print("总步数:",num_training_steps)
lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

#如果您可以访问GPU，请指定要使用GPU的设备。否则，在CPU上进行训练可能需要几个小时，而不是几分钟
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

#训练model

progress_bar = tqdm(range(num_training_steps))
#
# model.train()
# for epoch in range(num_epochs):
#     for step,batch in enumerate(train_dataloader):
#         batch = {k: v.to(device) for k, v in batch.items()}#字典{inputs_id:[],attention_mask:[],label:[]}
#         outputs = model(**batch)#[batch_size,sequence_len,num_labels]
#         loss = outputs.loss
#         loss.backward()
#
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)#更新已经处理一个批次了
#         if step % 100 == 0:
#             print(f'Step {step} / {num_training_steps} - Training loss: {loss}')
# #
# # 评估函数
#
metric = evaluate.load('seqeval')
#
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    for input_id,prediction,label in zip(batch['input_ids'], predictions,batch['labels']):
        index = input_id.tolist().index(102)#找到输入值为102的索引，102通常是终止标志[SEP]
        prediction2 = [ id2tag[t.item()]  for t in prediction[1:index]]#预测的：将中间的id转化为tag
        label2 = [ id2tag[t.item()]  for t in label[1:index]]#正确答案:
        metric.add(prediction=prediction2,  reference=label2)

results  = metric.compute() 
# print(results)

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
            outputs = model(**batch);

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        for input_id, prediction in zip(batch['input_ids'], predictions):
            index = input_id.tolist().index(102)
            sentence = tokenizer.decode(input_id[1:index]).replace(" ", "")
            prediction2 = [id2tag[t.item()] for t in prediction[1:index]]
            prediction_str = ' '.join(prediction2)

            line = f"{i}\u0001{sentence}\u0001{prediction_str}\n"
            file.write(line)
            i += 1