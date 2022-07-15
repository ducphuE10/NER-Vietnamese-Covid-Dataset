import torchtext


def get_data(path):
    train_data = []
    with open(path, encoding='utf-8') as f:
        words = []
        tags = []
        for line in f:
            line = line.strip()
            if not line:
                train_data.append([words, tags])
                words = []
                tags = []
            else:
                columns = line.split()
                words.append(columns[0])
                tags.append(columns[-1])
    return train_data


def read_file(path, data_fields, aug=False):
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
                # words.append(normalize_word(columns[0]))
                words.append(columns[0])
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


def get_instances_by_tag(dataset, tag):
    '''
        :param tag: ví dụ như JOB, ORGANIZATION, ....
        :return:
    '''
    # print(dataset)
    instances = []  # ví dụ job thì sẽ là: [[giáo viên], [công nhân], [y_tá, điều_dưỡng],...]
    for ex in dataset:
        tags = ex[1]
        for i in range(len(tags)):
            if tags[i] == f'B-{tag}':
                j = i + 1
                instance = [ex[0][i]]
                if j < len(tags):
                    while tags[j] == f'I-{tag}':
                        instance.append(ex[0][j])
                        j = j + 1
                        if j >= len(tags):
                            break

                if instance not in instances:
                    instances.append(instance)

    return instances

# examples = get_sent_by_tag('I-PATIENT_ID','PhoNER_COVID19/data/syllable/train_syllable.conll')
# # print(examples)
# for i in examples[0]:
#     print(i)

# print(len(examples[0][0]))
# print(len(examples[0][1]))


# dataset = get_data('dataset/test_word_origin.conll')
# for i in get_instances_by_tag(dataset,'NAME'):
#     if len(i) >1:
#         print(i)
