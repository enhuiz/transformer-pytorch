import json
import random


class Vocab():
    extra = ['<pad>', '<unk>', '<s>', '</s>', '<mask>']

    def __init__(self, sentences):
        self.vocab = []
        for sentence in sentences:
            self.vocab += sentence
        self.vocab = sorted(set(self.vocab))
        # extra at first so that different vocabs share the same extra idxs
        self.words = Vocab.extra + self.vocab
        self.inv_words = {w: i for i, w in enumerate(self.words)}

        assert self.words[:len(self.extra)] == self.extra

    @staticmethod
    def extra2idx(e):
        return Vocab.extra.index(e)

    def strip_beos_w(self, words):
        if words[0] == '<s>':
            del words[0]
        if words[-1] == '</s>':
            del words[-1]
        return words

    def word2idx(self, word):
        if word not in self.inv_words:
            word = '<unk>'
        return self.inv_words[word]

    def idx2word(self, idx):
        if idx < len(self):
            return self.words[idx]
        else:
            return '<unk>'

    def idxs2words(self, idxs):
        return [self.idx2word(idx) for idx in idxs]

    def words2idxs(self, words):
        return [self.word2idx(word) for word in words]

    def __len__(self):
        return len(self.words)

    def __str__(self):
        return ("Vocab(#words={}, #vocab={}, #extra={})\n"
                "Example: {}"
                .format(len(self), len(self.vocab),
                        len(self.extra),
                        random.sample(self.vocab,
                                      min(5, len(self.vocab)))))

    def __iter__(self):
        for w in self.words:
            yield w

    def dump(self, path):
        json.dump(self.__dict__, path)

    def load(self, path):
        self.__dict__.update(json.load(path))
