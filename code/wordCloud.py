from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
import csv
import jieba

STOP_WORDS_FILE_PATH = ',。'
filename = '../data/reviews1.csv'
file = pd.read_csv(filename, header=0)
text = [str(line) for line in file['review']]
word = ''
for i in text:
    word += i


def clean_data(text):
    word_dict = {}

    t = ''
    for w in text:
        t = t + str(w)
    word = jieba.lcut(t.replace('\n', ''))
    return word

cloud = WordCloud(font_path="simsun.ttf").generate(word)
plt.imshow(cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
