import torch
from dataset import Dataset
from model import lstm_crf
from train import Trainer
from torch.optim import Adam
import matplotlib.pyplot as plt


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

corpus = Dataset(train_path='./PhoNER_COVID19/data/word/train_word.conll',
                 val_path='./PhoNER_COVID19/data/word/dev_word.conll',
                 test_path='./PhoNER_COVID19/data/word/test_word.conll',
                 batch_size=64)

model = lstm_crf(
    word_input_dim=len(corpus.word_field.vocab),
    word_embedding_dim=300,
    char_embedding_dim=25,
    char_input_dim=len(corpus.char_field.vocab),
    char_cnn_filter_num=5,
    char_cnn_kernel_size=3,
    lstm_hidden_dim=64,
    output_dim=len(corpus.tag_field.vocab),
    lstm_layers=2,
    char_emb_dropout=0.5,
    word_emb_dropout=0.5,
    cnn_dropout=0.25,
    lstm_dropout=0.1,
    fc_dropout=0.25,
    word_pad_idx=corpus.word_pad_idx,
    char_pad_idx=corpus.char_pad_idx,
    tag_pad_idx=corpus.tag_pad_idx,
    use_char= True
)


model.init_embeddings()

# CRF transitions initialization for impossible transitions
model.init_crf_transitions(
        tag_list=corpus.tag_field.vocab.itos
)

print("Number of parameters: ",sum(p.numel() for p in model.parameters() if p.requires_grad))

trainer = Trainer(
    model=model,
    data=corpus,
    optimizer=Adam,
    device = device
)

history =  trainer.train(2)

print(history)
