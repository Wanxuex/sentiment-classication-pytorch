import os
import pandas as pd
import csv
def process(path):
    files = os.listdir(path)
    s = []
    neg, pos, neu = [], [], []
    label = []
    for file in files:  # 遍历文件夹
        f = pd.read_csv(path+'/'+file, header=0, index_col=0)
        s.extend(list(f['review']))
        label.extend(list(f['rating']))
        for i in range(len(f)):
            if int(f['rating'][i]) < 3:
                neg.append(f['review'][i])
            elif int(f['rating'][i]) > 3:
                pos.append(f['review'][i])
            else:
                neu.append(f['review'][i])
    pd.DataFrame({'review': neg}).to_csv("../data/neg.csv", encoding='utf-8')
    pd.DataFrame({'review': pos}).to_csv("../data/pos.csv", encoding='utf-8')
    pd.DataFrame({'review': neu}).to_csv("../data/neu.csv", encoding='utf-8')
    # save_file(neg, 'neg.txt')
    # save_file(pos, 'pos.txt')
    # save_file(neu, 'neu.txt')


def save_file(data, file):
    f1 = open('../data/'+file, 'w+', encoding='utf-8')
    for d in data:
        f1.write(d)
    f1.close()


process('../data/reviews')



