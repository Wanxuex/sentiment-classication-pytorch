import torch
import re
from torch.utils.data import Dataset
class CommentDataSet(Dataset):
    def __init__(self, data_path, word2ix, ix2word):
        self.data_path = data_path
        self.word2ix = word2ix
        self.ix2word = ix2word
        self.data, self.label = self.get_data_label()

    def __getitem__(self, idx: int):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)

    def get_data_label(self):
        data = []
        label = []
        with open(self.data_path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    label.append(torch.tensor(int(line[0]), dtype=torch.int64))
                except BaseException:  # 遇到首个字符不是标签的就跳过比如空行，并打印
                    print('not expected line:' + line)
                    continue
                line_words = re.split(r'[\s]', line)[1:-1]  # 按照空字符\t\n 空格来切分
                words_to_idx = []
                for w in line_words:
                    try:
                        index = self.word2ix[w]
                    except BaseException:
                        index = 0  # 测试集，验证集中可能出现没有收录的词语，置为0
                    #                 words_to_idx = [self.word2ix[w] for w in line_words]
                    words_to_idx.append(index)
                data.append(torch.tensor(words_to_idx, dtype=torch.int64))
        return data, label