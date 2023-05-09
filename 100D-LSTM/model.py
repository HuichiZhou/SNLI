# 导入深度学习框架torch
import torch
import torch.nn as nn


# 构建一个RNN模型
class Sentence_RNN(nn.Module):
    '''
        hidden_size: 隐藏层大小
        num_classes: 分类数量 对于SNLI数据集是三分类
        embeddings: 使用了哪个词嵌入 840B-300D
        embedding_size: 300D 
        num_layers: RNN网络的层数
    '''
    def __init__(self,
                 hidden_size,
                 num_classes,
                 embeddings,
                 embedding_size,
                 num_layers):
        super(Sentence_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        # 加载预训练词向量，并且冻结词向量
        self.emb = nn.Embedding.from_pretrained(embeddings=embeddings,
                                                freeze=True)
        # 分别对hypothesis 和 premise RNN进行编码
        self.rnn_hyp = nn.RNN(input_size=embedding_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True)
        self.rnn_prem = nn.RNN(input_size=embedding_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True)
        # fc1 200 * 500 的 神经网络构成
        self.fc1 = nn.Linear(2 * hidden_size, 500)
        # fc2 500 * 3 的 神经网络构成
        self.fc2 = nn.Linear(500, num_classes)
        # ReLU 为其激活函数
        self.activation = nn.ReLU()

    def forward(self, hyp_batch, premise_batch):
        """
            前向传播算法
        """
        # 对hypothesis 和 premise 加载预训练词向量进行词嵌入
        hyp_embedding_layer = self.emb(hyp_batch)
        prem_embedding_layer = self.emb(premise_batch)

        # 将隐藏层状态输出 在循环神经网络中我们一般将隐藏层状态提出进行后续计算
        hyp_out, hyp_hn = self.rnn_hyp(hyp_embedding_layer)
        prem_out, prem_hn = self.rnn_prem(prem_embedding_layer)
        
        # 隐藏层状态输出为 (num_layers * num_directions, batch_size, hidden_size)
        # 由于num_layers * num_direcitons = 1 * 1 所以下面的squeeze输出的结果为(batch_sizze, hidden_size)
        hyp_hn = torch.squeeze(hyp_hn)
        prem_hn = torch.squeeze(prem_hn)
        
        # 将隐藏层状态进行拼接
        combined_out = torch.cat((hyp_hn, prem_hn), dim=1)

        # 在使用fc1神经网络 + 激活函数
        second_last_out = self.activation(self.fc1(combined_out))
        # 最后一层神经网络进行输出
        out = self.fc2(second_last_out)
        return out


class Sentence_LSTM(nn.Module):
    '''
        hidden_size: 隐藏层大小
        num_classes: 分类数量 对于SNLI数据集是三分类
        embeddings: 使用了哪个词嵌入 840B-300D
        embedding_size: 300D 
        num_layers: RNN网络的层数
    '''

    def __init__(self,
                 hidden_size,
                 num_classes,
                 embeddings,
                 embedding_size,
                 num_layers):
        super(Sentence_LSTM, self).__init__()
        
        # 同上RNN
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.emb = nn.Embedding.from_pretrained(embeddings=embeddings,
                                                freeze=True)
        
        # 分别对hypothesis 和 premise LSTM进行编码
        # batch_first=True 即batch是第一个 
        self.lstm_hyp = nn.LSTM(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        self.lstm_prem = nn.LSTM(input_size=embedding_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 batch_first=True)
        
        # 如readme框架中的网络
        self.fc1 = nn.Linear(2 * hidden_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, num_classes)
        self.activation = nn.Tanh()

    def forward(self, hyp_batch, premise_batch):
        """
            前向传播算法
        """

        # 对hypothesis 和 premise 加载预训练词向量进行词嵌入
        hyp_embedding_layer = self.emb(hyp_batch)
        prem_embedding_layer = self.emb(premise_batch)

        # 使用lstm编码得到隐藏层
        hyp_out, (hyp_hn, hyp_cn) = self.lstm_hyp(hyp_embedding_layer)
        prem_out, (prem_hn, prem_cn) = self.lstm_prem(prem_embedding_layer)

        #同上
        hyp_hn = torch.squeeze(hyp_hn)
        prem_hn = torch.squeeze(prem_hn)

        #拼接
        combined_out = torch.cat((hyp_hn, prem_hn), dim=1)

        #前向传播框架
        first_layer = self.activation(self.fc1(combined_out))
        second_layer = self.activation(self.fc2(first_layer))
        third_layer = self.activation(self.fc3(second_layer))
        out = self.fc4(third_layer)
        return out