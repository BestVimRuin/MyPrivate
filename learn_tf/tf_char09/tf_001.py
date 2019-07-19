import codecs
import collections
from operator import itemgetter

RAW_DATA = r'D:\gitee\learn_tf\re_learn\data\simple-examples\data\ptb.train.txt'
counter = collections.Counter()
VOCAB_OUTPUT = 'ptb.vocab'
with codecs.open(RAW_DATA, 'r', 'utf-8') as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1

# 按照单次出现的次数 降序排列
sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]
sorted_words = ['<eos>'] + sorted_words

with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word+'\n')
