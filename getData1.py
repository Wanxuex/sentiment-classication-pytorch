import pandas as pd
import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
import random
import json
# 账号密码
def login(username, password):
    url = 'https://accounts.douban.com/passport/login?source=movie'
    header = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36',
        'Referer': 'https://accounts.douban.com/passport/login_popup?login_source=anony',
        'Origin': 'https://accounts.douban.com',
        'content-Type': 'application/x-www-form-urlencoded',
        'x-requested-with': 'XMLHttpRequest',
        'accept': 'application/json',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'zh-CN,zh;q=0.9',
        'connection': 'keep-alive'
        , 'Host': 'accounts.douban.com'
    }
    # 登陆需要携带的参数
    data = {
        'ck' : '',
        'name': '',
        'password': '',
        'remember': 'false',
        'ticket': ''
    }
    data['name'] = username
    data['password'] = password
    data = urllib.parse.urlencode(data)
    print(data)
    requests.packages.urllib3.disable_warnings()
    req = requests.post(url, headers=header, data=data, verify=False)
    cookies = requests.utils.dict_from_cookiejar(req.cookies)
    print(cookies)
    return cookies

def getNum(cookies):

    url = 'https://movie.douban.com/j/new_search_subjects?sort=U&range=0,10&tags=%E7%94%B5%E5%BD%B1&start=0&genres=%E5%89%A7%E6%83%85'
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',

        'x - client - data': 'CKm1yQEIh7bJAQimtskBCMS2yQEIqZ3KAQisx8oBCPXHygEItMvKAQijzcoBCJ / PygEI3NXKAQiTmssBCNCaywEIwpzLAQjVnMsB',
        'Decoded':'message ClientVariations {\
                        // Active client experiment variation IDs.\
                    repeated int32 variation_id = [3300009, 3300103, 3300134, 3300164, 3313321, 3318700, 3318773, 3319220, 3319459, 3319711, 3320540,\
                        3329299, 3329360, 3329602, 3329621]; }'
    }
    proxy = {'http': 'http://62.84.70.130:80'}
    requests.packages.urllib3.disable_warnings()
    req = requests.get(url, cookies=cookies, headers=header,  proxies=proxy)
    data = req.json()
    # f = open('movieid.json', 'w')
    # f.write(json.dumps(data))
    # f.close()
    products = data['data']
    numList = []
    for p in products:
        num = p['id']
        numList.append(num)
    return numList

    # 生成url
def generateURL(self, current_page):
    form = "电影"  # 形式
    feature = ""  # 特色
    type = "剧情"  # 类型
    countries = "美国"  # 地区
    start_year = "2000"  # 开始年限
    end_year = "2020"  # 截至年限
    # form = ""  # 形式
    # feature = ""  # 特色
    # type = ""  # 类型
    # countries = ""  # 地区
    # start_year = ""  # 开始年限
    # end_year = ""  # 截至年限

    url = "https://movie.douban.com/j/new_search_subjects?tags="
    if form != "" and form != None:
        url = url + form + ","

    if feature != "" and feature != None:
        url = url + feature + ","
    else:
        url = url[:-1]  # 如果feature为空则把最后一个逗号去掉

    url = url + "&start=" + str(current_page * 20)

    if type != "" and type != None:
        url = url + "&genres=" + type

    if countries != "" and countries != None:
        url = url + "&countries=" + countries

    if start_year != "" and start_year != None and end_year != "" and end_year != None:
        url = url + "&year_range=" + start_year + "," + end_year

    return url

def getcomment(mvid):
    start = 0
    all_products = []
    for i in range(12):

        if i != 0:
            time.sleep(random.randint(3,5))
        header = {
            'Host':'movie.douban.com',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
            'Connection': 'keep - alive',
            'Accept - Encoding': 'gzip, deflate, br',
            'Accept - Language': 'zh - CN, zh;q = 0.9 '
        }
        try:
            url = 'https://movie.douban.com/subject/'+str(mvid)+'/comments?start='+str(start) +'&limit=20&status=P&sort=new_score'
            start +=20
            requests.packages.urllib3.disable_warnings()
            req = requests.get(url, headers=header, verify=False)

            soup = BeautifulSoup(req.text, 'html.parser')
            products = soup.select('div.comment-item')
            for product in products:
                username = product.select('span.comment-info > a')[0].text
                vote = product.select('span.comment-vote')[0].span.text
                review = product.select('span.short')[0].text
                rating = product.select('span.comment-info')[0].find_all('span')[1].attrs['class'][0]
                if rating == 'allstar50':
                    rate = 5
                elif rating == 'allstar40' :
                    rate = 4
                elif rating == 'allstar30' :
                    rate = 3
                elif rating == 'allstar20' :
                    rate = 2
                elif rating == 'allstar10':
                    rate = 1
                all_products.append({
                    "username": username,
                    "vote":  vote,
                    "review": review,
                    "rating": rate
                })
        except Exception as e:  # 有异常退出
            print(e)
            break
    # 存储
    return all_products



if __name__ == '__main__':
    # cookies = login('13018013464', 'Wx2019123214')
    # time.sleep(3)
    # numl = getNum(cookies)
    # print(numl)

    numl = []
    df = pd.read_csv('data/movieid.csv', header=0)
    for p in df['0']:
        numl.append(p)
    print(numl)
    all_products = []

    for i in range(2, 20):
        # mvid = '25907124'
        mvid = numl[i]
        time.sleep(random.randint(15, 90))
        print("第" + str(i) + "个movie， id为：", mvid)
        i += 1
        all_products.extend(getcomment(mvid))
        keys = all_products[0].keys()
        pd.DataFrame(all_products, columns=keys).to_csv('data/reviews_{name}.csv'.format(name=mvid), encoding='utf-8')
        #
        # keys = all_products[0].keys()
        # pd.DataFrame(all_products, columns=keys).to_csv('reviews.csv', encoding='utf-8')
