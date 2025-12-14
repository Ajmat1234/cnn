import torch

class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=False):
        self.alphabet = alphabet + '-'  # blank at the end
        self.dict = {char: i + 1 for i, char in enumerate(alphabet)}
        self.ignore_case = ignore_case

    def encode(self, text):
        length = []
        result = []
        for item in text:
            item = item if not self.ignore_case else item.lower()
            length.append(len(item))
            for char in item:
                if char not in self.dict:
                    continue  # skip unknown chars
                result.append(self.dict[char])
        text = torch.IntTensor(result)
        length = torch.IntTensor(length)
        return text, length

    def decode(self, t, length, raw=False):
        if raw:
            return ''.join([self.alphabet[i - 1] for i in t if i > 0])
        texts = []
        index = 0
        for l in length:
            txt = []
            for i in range(l):
                char_idx = t[index + i]
                if char_idx > 0 and (i == 0 or t[index + i - 1] != char_idx):
                    txt.append(self.alphabet[char_idx - 1])
            texts.append(''.join(txt))
            index += l
        return texts
