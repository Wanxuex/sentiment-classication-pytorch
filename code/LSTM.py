import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, vocab, embedding_dim, hidden_size, num_layers=2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(len(vocab), embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,\
                            bidirectional=True, batch_first=True, dropout=0.1)
        self.fc = torch.nn.Linear(hidden_size * 2, 3)

    def forward(self, input):
        x = self.embeddings(input)
        x, (h_n, c_n) = self.lstm(x)
        output_fw = h_n[-2, :, :]  # 正向最后一次的输出
        output_bw = h_n[-1, :, :]  # 反向最后一次的输出
        output = torch.cat([output_fw, output_bw], dim=-1)
        out = self.fc(output)  # [batch_size, 2]
        # 可以考虑再添加一个全连接层作为输出层，激活函数处理。

        return F.log_softmax(out, dim=-1)


