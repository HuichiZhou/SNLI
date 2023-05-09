import torch
import torch.nn as nn
import torch.optim as optim

import torchtext.legacy as legacy
from model import Sentence_RNN, Sentence_LSTM

# 设置GPU设备加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 从legacy库得到 数据 和 正确分类
inputs = legacy.data.Field(lower=True, tokenize='spacy', batch_first=True)
answers = legacy.data.Field(sequential=False)

# 对数据集进行划分
train_data, validation_data, test_data = \
    legacy.datasets.SNLI.splits(text_field=inputs, label_field=answers)

# 构建词表
inputs.build_vocab(train_data, min_freq=1, vectors='glove.840B.300d')
answers.build_vocab(train_data)

# 得到预训练词向量
pretrained_embeddings = inputs.vocab.vectors

# 对于数据集进行迭代器
train_iterator, validation_iterator, test_iterator = \
    legacy.data.BucketIterator.splits((train_data, validation_data, test_data),
                                      batch_size=64,
                                      device=device)

# 超参数的设置
num_classes = 3
num_layers = 1
hidden_size = 100
embedding_size = 300

learning_rate = 0.001
batch_size = 64
num_epochs = 100


# 初始化神经网络
# model = Sentence_RNN(hidden_size,
#                      num_classes,
#                      inputs.vocab.vectors,
#                      embedding_size,
#                      num_layers).to(device)
model = Sentence_LSTM(hidden_size,
                      num_classes,
                      inputs.vocab.vectors,
                      embedding_size,
                      num_layers).to(device)

# 设置交叉熵损失函数 + 以及Adam优化 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 设置检查点 检查准确率
def check_accuracy(validation_iterator, model):
    num_correct = 0
    num_sample = 0

    with torch.no_grad():
        for val_batch_idx, val_batch in enumerate(validation_iterator):
            val_hyp = val_batch.hypothesis
            val_prem = val_batch.premise

            val_target = val_batch.label - 1
            scores = model(val_hyp, val_prem)
            # return the indices of each prediction
            _, predictions = scores.max(1)
            num_correct += (predictions == val_target).sum()
            num_sample += predictions.size(0)
        acc = (num_correct / num_sample)
        print("The val set accuracy is {}".format(acc))
    return acc


# 训练网络
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_iterator):
        model.train()
        # 得到数据
        hyp_sentences = batch.hypothesis
        prem_sentences = batch.premise

        # 得到标签
        target = batch.label - 1
        # 前向过程 并 得到损失
        scores = model(hyp_sentences, prem_sentences)
        loss = criterion(scores, target)

        # 反向传播过程
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

        # 梯度更新
        optimizer.step()

        model.eval()
        if batch_idx % 1000 == 0:
            print(loss)
            check_accuracy(validation_iterator, model)