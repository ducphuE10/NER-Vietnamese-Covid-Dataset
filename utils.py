import torchtext
from torchtext.data import Field, BucketIterator
import underthesea
from underthesea import word_tokenize

def read_file(path, datafields):
    with open(path, encoding='utf-8') as f:
        examples = []
        words = []
        tags = []
        for line in f:
            line = line.strip()
            if not line:
                examples.append(torchtext.data.Example.fromlist([words, tags], datafields))
                words = []
                tags = []
            else:
                columns = line.split()
                words.append(normalize_word(columns[0]))
                tags.append(columns[-1])
    return torchtext.data.Dataset(examples, datafields)


# nếu là số sẽ chuyển về 0, ví dụ covid-19 -> covid-00 hay 20-11-2001 -> 00-00-0000
def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word