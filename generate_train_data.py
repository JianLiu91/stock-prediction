import jieba

def all_news():
    date_news_dict = {}
    temp = ''
    with open('news_normal.txt') as file_input:
        for line in file_input:
            if line.startswith('\t'):
                d = line.strip()
                temp = d[:4] + '-' + d[4:6] + '-' + d[6:]
            else:
                date_news_dict.setdefault(temp, [])
                date_news_dict[temp].append(line.strip().replace('\t', ''))

    for elem in date_news_dict:
        # print elem, len(date_news_dict[elem])
        date_news_dict[elem] = date_news_dict[elem][:70]
        while len(date_news_dict[elem]) < 70:
            date_news_dict[elem].append('PAD')

    return date_news_dict


def all_stocks():
    stock_details = {}
    with open('stock_detail.txt') as file_in:
        for line in file_in:
            fields = line.strip().split('\t')
            sn, desc = fields[0], ' '.join(fields[1:])
            stock_details[sn] = desc

    all_stocks = []
    with open('stock_prices.txt') as file_in:
        for line in file_in:
            _, sn, dates, prices = line.strip().split('\t')
            all_stocks.append((sn, stock_details[sn], dates, prices))

    return all_stocks


news = all_news()
all_stocks = all_stocks()


for elem in all_stocks:
    sn, sd, dates, prices = elem
    dates = [str(x) for x in dates.split()]
    prices = [float(x) for x in prices.split()]

    t = ' '.join(jieba.cut(sd)).encode('utf8')

    span = 7
    for idx, d in enumerate(zip(dates, prices)):
        if idx + span >= len(prices): break
        date, price = d
        date_pre = dates[idx + span]
        price_pre = prices[idx + span]

        flag = 1 if price - price_pre > 0 else 0
        if date_pre in news:
            news_in_the_day = news[date_pre]
            news_in_the_day = [' '.join(jieba.cut(x)).encode('utf8') for x in news_in_the_day]
            print '\t'.join([sn, date, date_pre, str(flag), t, ' ||| '.join(news_in_the_day)])