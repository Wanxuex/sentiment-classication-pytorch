# -*- coding: utf-8 -*-
import pandas as pd
import jieba
import torch
import re
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from LSTM import LSTM
import time
from sklearn import metrics
from matplotlib import pyplot as plt

def get_data(path):
    f = pd.read_csv(path, header=0, index_col=0)
    review = f['review']
    label = f['label']
    return review, label

def stopwordslist(filepath):   # 定义函数创建停用词列表
    stopword = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]    #以行的形式读取停用词表，同时转换为列表
    return stopword

def clean_data(review):
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', '：', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '《', '》', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“',
                '\【', '\】', '\（', '\、', '。', '\）',  '\，', '[\s0-9a-zA-Z]']
    data = []
    for sentence in review:
        sentence = "".join(sentence)
        sentence = re.sub("|".join(fileters), " ", sentence)
        # result = [i for i in sentence.split(" ") if len(i) > 0]
        data.append(sentence)
    return data


def build_word_dict(text):
    words = []
    max_len = 0
    total_len = 0
    fileters = stopwordslist('../data/stopwords.txt')
    for line in text:
        # line = jieba.lcut(line)
        total_len += len(line)
        max_len = max(max_len, len(line))
        total_len += len(line)
        for w in line:
            if w not in fileters:
                words.append(w)
    words = list(set(words))#最终去重
    words = sorted(words) # 一定要排序不然每次读取后生成此表都不一致，主要是set后顺序不同
    #用unknown来表示不在训练语料中的词汇
    word2ix = {w:i+1 for i,w in enumerate(words)} # 第0是unknown的 所以i+1
    ix2word = {i+1:w for i,w in enumerate(words)}
    word2ix['<unk>'] = 0
    ix2word[0] = '<unk>'
    av_len = int(total_len/len(text))
    return word2ix, ix2word, max_len, av_len


def word_emmbeding(review, labels, word2ix, av_len):
    data = []
    label = []
    a = []
    for (line, l) in zip(review, labels):
        try:
            # label.append(torch.tensor(int(l), dtype=torch.int64))
            label.append(int(l))
        except BaseException:  # 遇到首个字符不是标签的就跳过比如空行，并打印
            print('not expected line:' + l)
            continue
        # line_words = re.split(r'[\s]', line)[1:-1]  # 按照空字符\t\n 空格来切分
        # line_words = jieba.lcut(line)
        a.append(len(line))
        line = transform(line, max_len=av_len)
        words_to_idx = []
        for w in line:
            try:
                index = word2ix[w]
            except BaseException:
                index = 0  # 测试集，验证集中可能出现没有收录的词语，置为0
            #                 words_to_idx = [self.word2ix[w] for w in line_words]
            words_to_idx.append(index)
        # data.append(torch.tensor(words_to_idx, dtype=torch.int64))
        data.append(words_to_idx)
    # plt.hist(a, bins=300, range=[0, 300], align='mid', density=True)
    # plt.title('Sentence length distribution graph')
    # plt.xlabel('Number of words')
    # plt.ylabel('Percentage')
    # plt.show()
    train, test, trainl, test_l = text_split(data, label, 0.2)
    return train, trainl, test, test_l

def transform( sentence, max_len=None):
    if max_len > len(sentence):
        sentence = sentence + [0] * (max_len - len(sentence)) # 填
    if max_len < len(sentence):
        sentence = sentence[:max_len] # 裁剪
    return sentence

def text_split(data,label, test_ratio):
    # 设置随机数种子，保证每次生成的结果都是一样的
    np.random.seed(42)
    np.random.shuffle(data)
    np.random.seed(42)
    np.random.shuffle(label)
    test_set_size = int(len(data) * test_ratio)
    return data[test_set_size:], data[:test_set_size], label[test_set_size:], label[:test_set_size]


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if 'is_training' in net.__code__.co_varnames:
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def _evaluate_acc_f1(data_loader, net,  device=None):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        net.eval()
        with torch.no_grad():
            for t_inputs, t_targets in data_loader:
                t_outputs = net(t_inputs)
                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1


def train(train_iter, test_iter, net, loss, optimizer, num_epochs):
    print("training。。。。 ")
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X
            y = y
            y_hat = net(X)
            l = loss(y_hat, y) # 交叉熵损失函数
            optimizer.zero_grad()
            l.backward()
            optimizer.step()# 优化方法
            train_l_sum += l.cpu().item()# 进入cpu中统计
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        # test_acc = evaluate_accuracy(test_iter, net)
        # print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
        #       % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        test_acc, test_f1 = _evaluate_acc_f1(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, test f1 %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, test_f1, time.time() - start))


class MyDataset(Dataset):
    def __init__(self, d, l):
        self.data = torch.tensor(d)
        self.label = torch.LongTensor(l)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    print('获取数据......')
    path = '../data/data_.csv'
    review, label = get_data(path)
    review = clean_data(review)
    print(len(label), len(review))
    neg, pos, neu = 0, 0, 0
    for l in label:
        if l==0:
            neg += 1
        elif l == 2:
            pos +=1
        else:
            neu += 1
    print('neg:', neg, 'pos:', pos, 'neu:', neu)
    # 绘制数据分布图
    # plt.bar(['neg', 'pos', 'neu'], [neg, pos, neu], align='center')
    # # 给条形图添加数据标注
    # for x, y in enumerate([neg, pos, neu]):
    #     plt.text(x-0.1, y+20, "%s" % y)
    # plt.title(' Data distribution graph')
    # plt.ylabel('count')
    # plt.xlabel('sentiment')
    # plt.show()

    review = [jieba.lcut(line) for line in review]
    word2ix, ix2word, max_len, av_len = build_word_dict(review)
    print("句子词数：", max_len, av_len)
    av_len = 125
    train_d, train_l, test, test_l = word_emmbeding(review, label, word2ix, av_len)

    train_iter = DataLoader(MyDataset(train_d, train_l), batch_size=64, shuffle=False)
    test_iter = DataLoader(MyDataset(test, test_l), batch_size=64, shuffle=False)
    model = LSTM(ix2word, 100, 100)
    lr, num_epochs = 0.01, 10
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss = nn.CrossEntropyLoss()
    train(train_iter, test_iter, model, loss, optimizer, num_epochs)







