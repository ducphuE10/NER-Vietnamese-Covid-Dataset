import torchtext
from random import randint
from augmentation import get_instances_by_tag, aug_replace_in_same_tag

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
    if aug:
        train_data = get_data(path)
        SYMPTOM_AND_DISEASE = get_instances_by_tag(train_data, 'SYMPTOM_AND_DISEASE')
        JOBS = get_instances_by_tag(train_data, 'JOB')

    with open(path, encoding='utf-8') as f:
        examples = []
        words = []
        tags = []
        for line in f:
            line = line.strip()
            if not line:

                '''AUGMENTATION'''
                if aug:
                    if "B-AGE" in tags and "I-AGE" not in tags and randint(0, 1):
                    # if "B-AGE" in tags and "I-AGE" not in tags:
                        b_age_index = tags.index("B-AGE")
                        i_age_instance = ["ngày", "tháng", "năm"]
                        i_age_insert = i_age_instance[random.randint(0, 2)]

                        words.insert(b_age_index + 1, i_age_insert)
                        tags.insert(b_age_index + 1, "I-AGE")

                    # AUG SYMPTOM_AND_DISEASE
                    # words_aug, tags_aug = aug_replace_in_same_tag(words, tags, 'SYMPTOM_AND_DISEASE',
                    #                                               SYMPTOM_AND_DISEASE)

                    # if words_aug:
                    #     # print(words_aug)
                    #     examples.append(torchtext.data.Example.fromlist([words_aug, tags_aug], data_fields))

                    words_aug, tags_aug = aug_replace_in_same_tag(words, tags, 'JOB', JOBS)

                    if words_aug and randint(0, 1):
                    # if words_aug:
                        examples.append(torchtext.data.Example.fromlist([words_aug, tags_aug], data_fields))

                '''END'''

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
