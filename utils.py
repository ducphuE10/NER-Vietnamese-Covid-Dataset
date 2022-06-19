import torchtext
import matplotlib.pyplot as plt

def read_file(path, data_fields):
    with open(path, encoding='utf-8') as f:
        examples = []
        words = []
        tags = []
        for line in f:
            line = line.strip()
            if not line:
                examples.append(torchtext.data.Example.fromlist([words, tags], data_fields))
                words = []
                tags = []
            else:
                columns = line.split()
                words.append(normalize_word(columns[0]))
                tags.append(columns[-1])
    return torchtext.data.Dataset(examples, data_fields)


# nếu là số sẽ chuyển về 0, ví dụ covid-19 -> covid-00 hay 20-11-2001 -> 00-00-0000
def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

def augment():
    pass

def get_sent_by_tag(tag,path):
    with open(path, encoding='utf-8') as f:
        examples = []
        words = []
        tags = []
        for line in f:
            line = line.strip()
            if not line:
                if tag in tags:
                    examples.append([words,tags])
                words = []
                tags = []
            else:
                columns = line.split()
                words.append(columns[0])
                tags.append(columns[-1])

    return examples

examples = get_sent_by_tag('I-AGE','./PhoNER_COVID19/data/word/train_word.conll')
print(examples)