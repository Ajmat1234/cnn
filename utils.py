import torch

class strLabelConverter(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.dict = {c: i + 1 for i, c in enumerate(alphabet)}  # blank=0 implicit

    def encode(self, texts):
        lengths = [len(text) for text in texts]
        result = []
        for text in texts:
            for char in text:
                if char in self.dict:
                    result.append(self.dict[char])
        return torch.IntTensor(result), torch.IntTensor(lengths)

    def decode(self, preds, lengths):
        texts = []
        index = 0
        for l in lengths:
            t = preds[index:index + l]
            char_list = []
            for i in range(l):
                if t[i] != 0 and (i == 0 or t[i] != t[i - 1]):
                    char_list.append(self.alphabet[t[i] - 1])
            texts.append(''.join(char_list))
            index += l
        return texts
