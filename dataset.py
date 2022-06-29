from torchtext.data import Field,NestedField, BucketIterator
from torchtext.vocab import Vocab
from utils import read_file
from collections import Counter
import torch


class Dataset:
    def __init__(self, train_path, val_path, test_path, batch_size, lower_word=True, wv_model=None, aug_train = True):
        self.word_field = Field(lower=lower_word)
        self.tag_field = Field(unk_token=None)

        self.char_nesting_field = Field(tokenize=list)
        self.char_field = NestedField(self.char_nesting_field)  # [batch_size, sent len, word len]
        self.data_fields = [(("word", "char"), (self.word_field, self.char_field)),
                            ("tag", self.tag_field)]

        self.train_dataset = read_file(train_path, self.data_fields, aug = aug_train)
        self.val_dataset = read_file(val_path, self.data_fields, aug = False)
        self.test_dataset = read_file(test_path, self.data_fields, aug = False)

        if wv_model:
            # retrieve word2vec model from gensim library
            # the file contains full word2vec model, not only key-vectors
            self.wv_model = wv_model
            self.embedding_dim = self.wv_model.vector_size
            # cannot create vocab with build_vocab(),
            # initiate vocab by building custom Counter based on word2vec model
            word_freq = {word: wv_model.get_vecattr(word, "count") for word in wv_model.index_to_key}
            word_counter = Counter(word_freq)
            self.word_field.vocab = Vocab(word_counter)
            # mapping each vector/embedding from word2vec model to word_field vocabs
            vectors = []
            for word, idx in self.word_field.vocab.stoi.items():
                if idx > 1:
                    vectors.append(torch.as_tensor(self.wv_model[word].tolist()))
                else:  # 0 is unk and 1 is pad
                    vectors.append(torch.zeros(self.embedding_dim))

            self.word_field.vocab.set_vectors(
                stoi=self.word_field.vocab.stoi,
                # list of vector embedding, orderred according to word_field.vocab
                vectors=vectors,
                dim=self.embedding_dim
            )

        else:
            self.word_field.build_vocab(self.train_dataset.word)

        # self.word_field.build_vocab(self.train_dataset.word)
        self.tag_field.build_vocab(self.train_dataset.tag)
        self.char_field.build_vocab(self.train_dataset.char)

        self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
            datasets=(self.train_dataset,
                      self.val_dataset,
                      self.test_dataset),
            batch_size=batch_size,
            sort=False)

        self.char_pad_idx = self.char_field.vocab.stoi[self.char_field.pad_token]
        self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
        self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]
