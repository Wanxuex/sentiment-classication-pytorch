import json
import pandas as pd

files = ['jsonfile/search_subjects.json', 'jsonfile/search_subjects (1).json', 'jsonfile/search_subjects (2).json', 'jsonfile/search_subjects (3).json']
for file in files:
    f = open(file, 'r', encoding='utf-8')
    ps = json.load(f)
    numl = []
    for p in ps['subjects']:
        num = p['id']
        numl.append(num)
    pd.DataFrame(numl).to_csv('data/movieid2.csv', mode='a', header=False, index=True, encoding='utf-8')
