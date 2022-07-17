import random
from random import randint

import visualize
from utils import get_data, get_instances_by_tag
from collections import Counter


def remove_position(position, list1):
    assert position < len(list1)
    if position == len(list1)-1:
        return list1[:-1]
    return list1[:position] + list1[position+1:]


def insert_position(position, list1, list2):
    return list1[:position] + list2 + list1[position:]


def replace_position(position, list1, list2, length):
    '''
    :param position: vị trí cần thay thế ở list1, ví dụ vị trí 1
    :param list1: ví dụ [anh, công_nhân, sửa_chữa, ống, nước, may_mắn]
    :param list2: list mà thay thế vị trí cần chèn
    :param length:
    :return:
    '''
    return list1[:position] + list2 + list1[position + length:]


def insert_I_AGE(word, tag):
    if "B-AGE" in tag and "I-AGE" not in tag:
        b_age_index = tag.index("B-AGE")
        i_age_instance = ["ngày", "tháng", "năm"]
        i_age_insert = i_age_instance[random.randint(0, 2)]

        word[b_age_index] = randint(1, 80)
        word.insert(b_age_index + 1, i_age_insert)
        tag.insert(b_age_index + 1, "I-AGE")
    return word, tag


def replace_in_same_tag(word, tag, tag_aug, tag_instances):
    '''
    :param tag_instances:
    :param word: list of words in sent []
    :param tag: list of tag in sent []
    :param tag_aug: tag_need to aug
    :return: [words,tags] after replace
    '''
    words_aug = word
    tags_aug = tag

    if f"B-{tag_aug}" in tag:
        b_indexes = [i for i, x in enumerate(tag) if x == f'B-{tag_aug}']
        for i in range(len(b_indexes)):
            b_index = b_indexes[i]
            i_index = b_index + 1
            while tags_aug[i_index] == f'I-{tag_aug}':
                i_index = i_index + 1
                if i_index >= len(tags_aug):
                    break

            index_replace = randint(0, len(tag_instances) - 1)
            tag_replace = tag_instances[index_replace]

            words_aug = replace_position(b_index, words_aug, tag_replace, i_index - b_index)
            tags_aug = replace_position(b_index, tags_aug, [f'B-{tag_aug}'] + (len(tag_replace) - 1) * [f'I-{tag_aug}'],
                                        i_index - b_index)

            for j in range(i, len(b_indexes) - 1):
                b_indexes[j + 1] += len(tag_replace) - (i_index - b_index)

        return words_aug, tags_aug

    else:
        return word, tag


def random_delete_outside(word, tag, num_delete=1):
    word_aug = word
    tag_aug = tag
    for i in range(num_delete):
        O_indexes = [j for j in range(len(tag_aug)) if tag_aug[j] == 'O']
        if len(O_indexes) == 0:
            break
        index = O_indexes[randint(0, len(O_indexes) - 1)]

        word_aug = remove_position(index, word_aug)
        tag_aug = remove_position(index, tag_aug)

    return word_aug,tag_aug



'''

train_data = get_data('dataset/train_word_update.conll')
JOBS = get_instances_by_tag(train_data, 'JOB')
NAME = get_instances_by_tag(train_data, 'NAME')
TRANSPORT = get_instances_by_tag(train_data, 'TRANSPORTATION')


train_data_aug = []
for i,(word,tag) in enumerate(train_data):
    if 'I-PATIENT_ID' in tag or 'I-NAME' in tag:
        for j in range(10):
            word_aug, tag_aug = random_delete_outside(word, tag, random.randint(1, 4))
            train_data_aug.append([word_aug, tag_aug])

    if randint(0, 1):
        word,tag = insert_I_AGE(word,tag)
    if 'B-JOB' in tag or 'B-TRANSPORTATION' in tag:
        if 'B-LOCATION' not in tag:
            for j in range(10):
                word_aug, tag_aug = replace_in_same_tag(word, tag, 'JOB', JOBS)
                word_aug, tag_aug = replace_in_same_tag(word_aug, tag_aug, 'TRANSPORTATION', TRANSPORT)
                word_aug, tag_aug = random_delete_outside(word_aug, tag_aug, random.randint(1,4))
                # word_aug, tag_aug = insert_I_AGE(word_aug, tag_aug)
                assert len(word_aug) == len(tag_aug)
                train_data_aug.append([word_aug,tag_aug])


train_data.extend(train_data_aug)

with open('dataset/train_word_aug_phu.conll','w',encoding='utf-8') as f:
    for word,tag in train_data:
        assert len(word) == len(tag)
        for i in range(len(word)):
            f.write(f'{word[i]} {tag[i]}\n')
        f.write('\n')

'''
