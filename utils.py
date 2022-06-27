import torchtext
from random import randint

def read_file(path, data_fields, aug = False):
    with open(path, encoding='utf-8') as f:
        examples = []
        words = []
        tags = []
        for line in f:
            line = line.strip()
            if not line:
                #Augment for I-AGE
                if aug == True:
                    words_aug, tags_aug = 0,0



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


def extract_instances_of_tag(tag, path):
    pass


def get_sent_by_tag(tag, path):
    with open(path, encoding='utf-8') as f:
        examples = []
        words = []
        tags = []
        for line in f:
            line = line.strip()
            if not line:
                if tag in tags:
                    examples.append([words, tags])
                words = []
                tags = []
            else:
                columns = line.split()
                words.append(columns[0])
                tags.append(columns[-1])

    return examples

# examples = get_sent_by_tag('I-PATIENT_ID','PhoNER_COVID19/data/syllable/train_syllable.conll')
# # print(examples)
# for i in examples[0]:
#     print(i)

# print(len(examples[0][0]))
# print(len(examples[0][1]))
