import numpy
import torch
from dataset import Dataset
from model import BiLSTM
from train import Trainer
import torch.nn as nn
from torch.optim import Adam


if __name__ == '__main__':
    corpus = Dataset(train_path='./PhoNER_COVID19/data/word/train_word.conll',
                    val_path='./PhoNER_COVID19/data/word/dev_word.conll',
                    test_path='./PhoNER_COVID19/data/word/test_word.conll',
                    batch_size=64)

    print(f"Train set: {len(corpus.train_dataset)} sentences")
    print(f"Val set: {len(corpus.val_dataset)} sentences")
    print(f"Test set: {len(corpus.test_dataset)} sentences")

    bilstm = BiLSTM(
        input_dim=len(corpus.word_field.vocab),
        embedding_dim=300,
        hidden_dim=64,
        output_dim=len(corpus.tag_field.vocab),
        lstm_layers=1,
        word_pad_idx=corpus.word_pad_idx
    )
    bilstm.init_weights()
    bilstm.init_embeddings(word_pad_idx=corpus.word_pad_idx)
    print(f"The model has {bilstm.count_parameters():,} trainable parameters.")
    print(bilstm)

    ner = Trainer(
        model=bilstm,
        data=corpus,
        optimizer_cls=Adam,
        loss_fn_cls=nn.CrossEntropyLoss
    )
    train_losses, val_losses, train_accuracies, val_accuracies = ner.train(3)