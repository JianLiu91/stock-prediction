import tushare as ts
import pandas as pd
import requests
from bs4 import BeautifulSoup

def sample_stocks():
    df = pd.read_csv('stock_no.txt', header=None, dtype=str)
    industry_group = df.groupby(3)

    for elem in industry_group:
        name, stocks = elem
        t = stocks[1].sample(5)

        for e in t:
            print name + '\t' + str(e)


def get_price(no):
    df = ts.get_hist_data(str(no))['close']
    prices = df.tolist()
    dates = df.index.tolist()

    return [str(x) for x in dates], [str(x) for x in prices]


def stocks_price():
    with open('sample_stocks.txt') as file_in:
        for line in file_in:
            name, sn = line.strip().split('\t')
            dates, prices = get_price(sn)
            print '\t'.join([name, sn, ' '.join(dates), ' '.join(prices)])


def get_company_detail(stock_no):
    url = 'http://stock.jrj.com.cn/share,' + stock_no + ',gsgk.shtml'
    content = requests.get(url)
    content.encoding = 'gbk'
    soup = BeautifulSoup(content.text, 'lxml')
    columns = soup.find('td', class_='m').findAll('td')
    temp = columns[-2].text + columns[-1].text
    return temp.replace('\n', '')


def stocks_detail():
    with open('stock_no.txt') as file_in:
        for line in file_in:
            _, stock_no, a, b = line.strip().split(',')
            desc = get_company_detail(stock_no)
            desc = desc.encode('utf8')
            print '\t'.join([stock_no, a, b, desc])

# stocks_detail()

# stocks_price()

# df = get_price('603369')
# print df