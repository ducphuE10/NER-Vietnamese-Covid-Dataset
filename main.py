import torch
from dataset import Dataset
from model import lstm_crf
from train import Trainer
from torch.optim import Adam

level = 'word'
device = 'cuda:0'
w2v_path = 'word2vec_vi_words_300dims/word2vec_vi_words_300dims.txt'
# wv_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path)

corpus = Dataset(train_path=f'dataset/train_{level}_update.conll',
                 val_path=f'dataset/dev_{level}.conll',
                 test_path=f'dataset/test_{level}.conll',
                 batch_size=36,
                 lower_word = True,
                 wv_model = None
                 )

# bar_chart(corpus.train_dataset)

model = lstm_crf(
    word_input_dim=len(corpus.word_field.vocab),
    word_embedding_dim=300,
    char_embedding_dim=50,
    char_input_dim=len(corpus.char_field.vocab),
    char_cnn_filter_num=5,
    char_cnn_kernel_size=3,
    lstm_hidden_dim=100,
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


# model.init_embeddings(
#     pretrain=corpus.word_field.vocab.vectors,
#     freeze=True
# )

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
    device = device,
    path='pretrain/model.pt'
)

history =  trainer.train(5)

print(history)

