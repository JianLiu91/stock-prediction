import datetime
import time
import requests

from bs4 import BeautifulSoup

start = datetime.datetime(2015, 1, 1)  # year, month, day
end = datetime.datetime(2017, 5, 7)  # year, month, day

# only edit these if you're having problems
delay = 1  # time to wait on each page load before reading the page

days = (end - start).days + 1

def format_day(date):
    day = '0' + str(date.day) if len(str(date.day)) == 1 else str(date.day)
    month = '0' + str(date.month) if len(str(date.month)) == 1 else str(date.month)
    year = str(date.year)
    return ''.join([year, month, day])

def form_url(until):
    # p1 = 'http://finance.jrj.com.cn/biz/xwk/' + until[:6] + '/' + until
    p1 = 'http://finance.jrj.com.cn/xwk/' + until[:6] + '/' + until
    p2 =  '_1.shtml'
    return p1 + p2

def increment_day(date, i):
    return date + datetime.timedelta(days=i)


def parser_content(url_content):
    soup = BeautifulSoup(url_content, 'lxml')
    try:
        news_list = soup.find('div', class_= 'main').find('ul')
    except:
        return
    for elem in news_list.findAll('li'):
        try:
            print elem.findAll('a')[1].text.strip()
        except:
            pass

for day in range(days):

    d1 = format_day(increment_day(start, 0))
    d2 = format_day(increment_day(start, 1))
    url = form_url(d1)

    print('\t' + d1)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.80 Safari/537.36'
    }

    content = requests.get(url, headers=headers)

    parser_content(content.text)


    time.sleep(1)
    start = increment_day(start, 1)