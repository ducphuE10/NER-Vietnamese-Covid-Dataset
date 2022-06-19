from torchtext.data import Field,NestedField, BucketIterator
from utils import read_file


class Dataset:
    def __init__(self, train_path, val_path, test_path, batch_size):
        self.word_field = Field(lower=True)
        self.tag_field = Field(unk_token=None)

        self.char_nesting_field = Field(tokenize=list)
        self.char_field = NestedField(self.char_nesting_field)  # [batch_size, sent len, word len]
        self.data_fields = [(("word", "char"), (self.word_field, self.char_field)),
                            ("tag", self.tag_field)]

        self.train_dataset = read_file(train_path, self.data_fields)
        self.val_dataset = read_file(val_path, self.data_fields)
        self.test_dataset = read_file(test_path, self.data_fields)

        self.word_field.build_vocab(self.train_dataset.word)
        self.tag_field.build_vocab(self.train_dataset.tag)
        self.char_field.build_vocab(self.train_dataset.char)

        self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
                                                                        datasets=(self.train_dataset,
                                                                                  self.val_dataset,
                                                                                  self.test_dataset),
                                                                        batch_size=batch_size,
                                                                        sort= False)

        self.char_pad_idx = self.char_field.vocab.stoi[self.char_field.pad_token]
        self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
        self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]
