import torch
import torch.nn as nn


class encoder(nn.Module):
    """
     接下来按照论文中使用BiLSTM对假设进行编码
     hidden_size:隐藏层的大小。
     embeddings:一个预先训练好的嵌入矩阵，用于将单词编码为密集向量。
     embedding_size:嵌入向量的大小。
     num_layers:BiLSTM 输出层的数量。
     p:dropout 概率，防止过拟合。
     dropout:一个 PyTorch 中的 nn.Dropout 模块，用于在神经网络的训练过程中随机丢弃一定比例的节点。
     emb:一个 PyTorch 中的 nn.Embedding 模块，用于将单词编码为密集向量。
     bilstm_layer:一个 PyTorch 中的 nn.LSTM 模块，实现了一个双向的 LSTM。
    """

    def __init__(self, hidden_size, embeddings, embedding_size, num_layers, p):
        super(encoder, self).__init__()
        self.hidden_size = hidden_size #隐藏层大小
        self.dropout = nn.Dropout(p)  #丢弃层
        #加载预训练词向量，导入embedding，冻结参数
        self.emb = nn.Embedding.from_pretrained(
            embeddings=embeddings, freeze=True
        )
        #加载BiLSTM
        self.bilstm_layer = nn.LSTM(
            input_size=embedding_size, #词向量维度
            hidden_size=hidden_size, #隐藏层维度
            num_layers=num_layers, #输出层维度
            batch_first=True, #第一个维度是否是批次大小
            dropout=p, #丢弃率
            bidirectional=False, #是否为双向lstm
        )

    def forward(self, sentence_batch, sequence_length):
        """
        Feed word embedding batch into LSTM
        """
        sentence_embedding_layer = self.dropout(self.emb(sentence_batch)) #将输入进行编码 然后随机丢弃结点
        #打包一个边长序列
        packed_embeddings = nn.utils.rnn.pack_padded_sequence( 
            sentence_embedding_layer,
            sequence_length.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        #将其丢进lstm模型中输出其
        """
        out: 输出值
        hn: 隐藏层值
        cn: 记忆层值
        """
        out, (hn, cn) = self.bilstm_layer(packed_embeddings)
        #这是与上面代码的逆操作，使得其边长一个变长序列
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        return out, hn, cn


class inner_attention(nn.Module):
    """
    内 注意力机制
    """

    def __init__(self, encoder, p=0.25):
        super(inner_attention, self).__init__()
        self.encoder = encoder
        hidden_size = self.encoder.hidden_size
        self.WY = nn.Linear(hidden_size, hidden_size) #论文中的W^y的权重 W^y * Y
        self.WH = nn.Linear(hidden_size, hidden_size) #论文中的W^h的权重 W^h * R_ave
        self.W = nn.Linear(hidden_size, 1) #论文中w^t w^t * M  M = tanh(W^y * Y + W^h * R_ave * e_L)
        self.fc1 = nn.Linear(4 * hidden_size, 300) # 8 * hidden_size , embbeding_size
        self.fc2 = nn.Linear(300, 300) # 线性层
        self.fc3 = nn.Linear(300, 3) # 输出层
        self.tanh = nn.Tanh() #激活函数
        self.softmax_layer = nn.Softmax(dim=1) #softmax函数
        self.dropout = nn.Dropout(p) #丢弃函数

        """
        初始化权重
        """
        self.WY.weight.data = nn.init.xavier_uniform_(self.WY.weight.data)
        self.WH.weight.data = nn.init.xavier_uniform_(self.WH.weight.data)
        self.W.weight.data = nn.init.xavier_uniform_(self.W.weight.data)

        self.fc1.weight.data = nn.init.xavier_uniform_(self.fc1.weight.data)
        self.fc2.weight.data = nn.init.xavier_uniform_(self.fc2.weight.data)
        self.fc3.weight.data = nn.init.xavier_uniform_(self.fc3.weight.data)
        #层归一化函数
        self.layernorm = nn.LayerNorm(normalized_shape=300)

    def forward(self, prem_batch, hyp_batch, prem_length, hyp_length):
        """
            prem_batch:  premise sentence indexes
            hyp_batch:   hypothesis sentence indexes
            prem_length: premise sentence length
            hyp_length:  hypothesis sentence length
        """
        #对premise进行编码 返回输出参数 隐藏层 记忆层
        prem_bilstm_output, _, _ = self.encoder(prem_batch, prem_length)
        #对hypothesis进行编码
        hyp_bilstm_output, _, _ = self.encoder(hyp_batch, hyp_length)
        # 形状为 batch * sequence * embedding
        prem_mean_vec = torch.mean(prem_bilstm_output, dim=1, keepdim=True)
        hyp_mean_vec = torch.mean(hyp_bilstm_output, dim=1, keepdim=True)

        # 形状为 batch * embedding * sequence
        prem_mean_vec = prem_mean_vec.permute(0, 2, 1)
        hyp_mean_vec = hyp_mean_vec.permute(0, 2, 1)

        # M = tanh(W^y * Y + W^h * (R_ave · e_L))
        M_premise = self.WY(prem_bilstm_output) + self.WH(
            torch.matmul(
                prem_mean_vec,
                torch.ones([1, prem_bilstm_output.shape[1]], device="cuda"),
            ).permute(0, 2, 1)
        )
        
        # batch * sequence * embedding 对其进行激活
        M_premise = self.tanh(M_premise)
        # Size of weights premise is like batch * sequence
        weights_premise = self.softmax_layer(self.W(M_premise).squeeze(2))

        weighted_premise = torch.bmm(
            M_premise.permute(0, 2, 1), weights_premise.unsqueeze(2)
        )
        weighted_premise = weighted_premise.squeeze(2)

        
        #同理
        M_hyp = self.WY(hyp_bilstm_output) + self.WH(
            torch.matmul(
                hyp_mean_vec,
                torch.ones([1, hyp_bilstm_output.shape[1]], device="cuda"),
            ).permute(0, 2, 1)
        )
        # batch * sequence * embedding
        M_hyp = self.tanh(M_hyp)
        # batch * sequence
        weights_hyp = self.softmax_layer(self.W(M_hyp).squeeze(2))
        weighted_hyp = torch.bmm(
            M_hyp.permute(0, 2, 1), weights_hyp.unsqueeze(2)
        )
        weighted_hyp = weighted_hyp.squeeze(2)

        # 减
        sentence_difference = weighted_premise - weighted_hyp
        # 乘
        sentence_multiplication = weighted_premise * weighted_hyp
        # 拼接
        sentence_matching = torch.cat(
            (
                weighted_premise,
                sentence_multiplication,
                sentence_difference,
                weighted_hyp,
            ),
            dim=1,
        )

        fc1_layer = self.fc1(sentence_matching)
        fc1_layer = self.tanh(fc1_layer)
        fc1_layer = self.layernorm(fc1_layer)
        fc1_layer = self.dropout(fc1_layer)

        fc2_layer = self.fc2(fc1_layer)
        fc2_layer = self.tanh(fc2_layer)
        fc2_layer = self.layernorm(fc2_layer)
        fc2_layer = self.dropout(fc2_layer)

        out = self.fc3(fc2_layer)
        return out