from random import randint

def insert_position(position, list1, list2):
    return list1[:position] + list2 + list1[position:]

def replace_position(position, list1, list2, length):
    return list1[:position] + list2 + list1[position + length :]

def aug_i_age(train_data):
    for i in range(len(train_data)):
        tags = train_data[i][1]
        if "B-AGE" in tags and "I-AGE" not in tags and randint(0, 1):
            b_age_index = tags.index("B-AGE")
            train_data[i][0].insert(b_age_index + 1, "tháng")
            train_data[i][1].insert(b_age_index + 1, "I-AGE")
            # print(ex[0])

    return train_data


def aug_I_AGE_sent(words, tags, i_age_list):
    if "B-AGE" in tags and "I-AGE" not in tags and randint(0, 1):
        b_age_index = tags.index("B-AGE")
        words.insert(b_age_index + 1, "tháng")
        tags.insert(b_age_index + 1, "I-AGE")

    return words, tags


def aug_replace_in_same_tag(words, tags, tag_aug, tag_instances):
    '''
    :param tag_instances:
    :param words: list of words in sent []
    :param tags: list of tag in sent []
    :param tag_aug: tag_need to aug
    :return: [words,tags] after replace
    '''
    words_aug = words
    tags_aug = tags
    # if f"B-{tag_aug}" in tags and randint(0, 1):
    if f"B-{tag_aug}" in tags:
        b_indexes = [i for i, x in enumerate(tags) if x == f'B-{tag_aug}']
        # print(len(b_indexes))
        for i in range(len(b_indexes)):
            b_index = b_indexes[i]
            i_index = b_index + 1
            while tags_aug[i_index] == f'I-{tag_aug}':
                i_index = i_index + 1
                if i_index >= len(tags_aug):
                    break

            index_replace = randint(0, len(tag_instances) - 1)
            tag_replace = tag_instances[index_replace]
            # print(tag_replace)
            words_aug = replace_position(b_index, words_aug, tag_replace,i_index- b_index)
            tags_aug = replace_position(b_index,tags_aug, [f'B-{tag_aug}'] + (len(tag_replace)-1)*[f'I-{tag_aug}'],i_index- b_index)

            for j in range(i,len(b_indexes)-1):
                b_indexes[j+1] += len(tag_replace) - (i_index- b_index)

        # print("words before replace:", ' '.join(words))
        # print("words after replace:", ' '.join(words_aug))
        # print(list(zip(words_aug, tags_aug)))
        # print("===============================")
        return words_aug, tags_aug

    else:
        return 0,0


def aug_JOB_sent_insert_random(words, tags, job_list):
    if "B-NAME" in tags and "B-JOB" not in tags and randint(0, 1):
        b_name_index = tags.index("B-NAME")
        words.insert(b_age_index + 1, "tháng")
        tags.insert(b_age_index + 1, "I-AGE")

    return words, tags

#
# SYMPTOM_AND_DISEASE = get_instances_by_tag(dataset, 'SYMPTOM_AND_DISEASE')
# JOBS = get_instances_by_tag(dataset, 'JOB')
# #
# # # print(len(dataset))
# for ex in dataset[:]:
#     words = ex[0]
#     tags = ex[1]
#
#     words_aug,tags_aug = aug_replace_in_same_tag(words, tags, 'SYMPTOM_AND_DISEASE', SYMPTOM_AND_DISEASE)
#     if words_aug:
#         # print(words_aug)
#         dataset.append([words_aug,tags_aug])
#
#     words_aug, tags_aug = aug_replace_in_same_tag(words, tags, 'JOB', JOBS)
#     if words_aug:
#         dataset.append([words_aug, tags_aug])

