import numpy as np


class WordSequence(object):
    PAD_TAG = "<pad>"
    UNK_TAG = "<unk>"
    START_TAG = "<s>"
    END_TAG = "</S>"

    PAD = 0
    UNK = 1
    START = 2
    END = 3

    def __init__(self):
        self.dict = {
            WordSequence.PAD_TAG: WordSequence.PAD,
            WordSequence.UNK_TAG: WordSequence.UNK,
            WordSequence.START_TAG: WordSequence.START,
            WordSequence.END_TAG: WordSequence.END,
        }
        self.fited = False

    def to_index(self, word):
        assert self.fited, "WordSequence 尚未进行 fit 操作"
        if word in self.dict:
            return self.dict[word]
        return WordSequence.UNK

    def to_word(self, index):
        assert self.fited, "WordSequence 尚未进行 fit 操作"
        for k, v in self.dict.items():
            if v == index:
                return k
        return WordSequence.UNK_TAG

    def size(self):
        assert self.fited, "WordSequence 尚未进行 fit 操作"
        return len(self.dict) + 1

    def __len__(self):
        return self.size()

    def fit(self, sentences, min_count=5, max_count=None, max_features=None):
        assert not self.fited, "WordSequence 只能fit一次"
        count = {}
        # 词频统计
        for sentence in sentences:
            arr = list(sentence)
            for a in arr:
                if a not in count:
                    count[a] = 0
                count[a] += 1

        if min_count is not None:
            count = {k: v for k, v in count.items() if v >= min_count}
        if max_count is not None:
            count = {k: v for k, v in count.items() if v <= max_count}
        self.dict = {
            WordSequence.PAD_TAG: WordSequence.PAD,
            WordSequence.UNK_TAG: WordSequence.UNK,
            WordSequence.START_TAG: WordSequence.START,
            WordSequence.END_TAG: WordSequence.END,
        }

        # isinstance 判断类型， 判断max_features(最大特征数) 是否为整型
        if isinstance(max_features, int):
            count = sorted(list(count.items()), key=lambda x: x[1])
            if max_features is not None and len(count) > max_features:
                count = count[-int(max_features):]
            for w, _ in count:
                self.dict[w] = len(self.dict)
        else:
            for w in sorted(count.keys()):
                self.dict[w] = len(self.dict)
        self.fited = True

    def transform(self, sentence, max_len=None):
        """Sentence2Vec"""
        assert self.fited, "WordSequence 尚未进行fit操作"
        if max_len is not None:
            r = [self.PAD] * max_len
        else:
            r = [self.PAD] * len(sentence)

        for index, a in enumerate(sentence):
            if max_len is not None and index >= len(r):
                break
            r[index] = self.to_index(a)

        return np.array(r)

    def inverse_transform(self, index, ignore_pad=False, ignore_unk=False, ignore_start=False, ignore_end=False):
        """Vec2Sentence"""
        ret = []
        for i in index:
            word = self.to_word(i)
            if word == WordSequence.PAD_TAG and ignore_pad:
                continue
            if word == WordSequence.UNK_TAG and ignore_unk:
                continue
            if word == WordSequence.START_TAG and ignore_start:
                continue
            if word == WordSequence.END_TAG and ignore_end:
                continue
            ret.append(word)
        return ret


def test():
    ws = WordSequence()
    ws.fit([
        ['你', '好', '啊'],
        ['你', '好', '哦'],
        ['你', '是', '谁', '啊'],
        ['你', '好', '嘛'],
        ['你', '是', '哪', '位'],
        ['我', '们', '是', '冠军'],
    ])
    index = ws.trainsform(['我', '们'])
    print(index)
    back = ws.inverse_transform(index)
    print(back)


if __name__ == '__main__':
    # test()
    print("123")
