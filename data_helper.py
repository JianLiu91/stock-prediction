import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn


vocab_processor = learn.preprocessing.VocabularyProcessor(100)


def generate_t():
    word_dict = {}
    with open('train_data.txt') as file_in:
        for line in file_in:
            _, _, _, label, desc, news =  line.strip().split('\t')
            news = news.split(' ||| ')
            for elem in desc.split(' '):
                word_dict.setdefault(elem, 0)
                word_dict[elem] += 1
            for x in news:
                for t in x.split(' '):
                    word_dict.setdefault(t, 0)
                    word_dict[t] += 1

    for k in word_dict:
        print k + '\t' + str(word_dict[k])


def fillinwv(words, word_dict, length):
    temp = [word_dict['PAD']] * length
    words = words.split()
    for idx, word in enumerate(words):
        if idx >= len(temp):
            break
        if not word in word_dict:
            temp[idx] = 0
        else:
            temp[idx] = word_dict[word]
    return temp


def read_data():
    word_dict = {}
    word_dict['UNK'] = 0
    word_dict['PAD'] = 1

    with open('word_dict.txt') as file_in:
        for line in file_in:
            try:
                word, _ = line.strip().split('\t')
                word_dict[word] = len(word_dict)
            except:
                pass

    # print len(word_dict)
    examples = []
    with open('train_data.txt') as file_in:
        for line in file_in:
            _, _, _, label, desc, news =  line.strip().split('\t')
            news = news.split(' ||| ')
            desc = fillinwv(desc, word_dict, 150)
            news_all = []
            for elem in news:
                temp = fillinwv(elem, word_dict, 20)
                news_all += temp
            examples.append((label, desc, news_all))

    x = [[] + elem[1] + elem[2] for elem in examples]
    y = map(lambda t: int(t[0]), examples)
    y = [[1, 0] if t == 0 else [0, 1] for t in y]

    return np.asarray(x), np.asarray(y), len(word_dict)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# x, y, vocab_size = read_data()
# print x