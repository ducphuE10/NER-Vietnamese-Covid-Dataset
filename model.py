import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

class BiLSTM(nn.Module):
  def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, lstm_layers, word_pad_idx):
    super().__init__()
    self.embedding_dim = embedding_dim
    # LAYER 1: Embedding
    self.embedding = nn.Embedding(
        num_embeddings=input_dim,
        embedding_dim=embedding_dim,
        padding_idx=word_pad_idx
    )

    # LAYER 2: BiLSTM
    self.lstm = nn.LSTM(
        input_size=embedding_dim,
        hidden_size=hidden_dim,
        num_layers=lstm_layers,
        bidirectional=True
    )
    # LAYER 3: Fully-connected
    self.fc = nn.Linear(hidden_dim * 2, output_dim)  # times 2 for bidirectional

  def forward(self, sentence):
    # sentence = [sentence length, batch size]
    # embedding_out = [sentence length, batch size, embedding dim]
    embedding_out = self.embedding(sentence)
    # lstm_out = [sentence length, batch size, hidden dim * 2]
    lstm_out, _ = self.lstm(embedding_out)
    # ner_out = [sentence length, batch size, output dim]
    ner_out = self.fc(lstm_out)
    return ner_out

  def init_weights(self):
    # to initialize all parameters from normal distribution
    # helps with converging during training
    for name, param in self.named_parameters():
      nn.init.normal_(param.data, mean=0, std=0.1)

  def init_embeddings(self, word_pad_idx):
    # initialize embedding for padding as zero
    self.embedding.weight.data[word_pad_idx] = torch.zeros(self.embedding_dim)

  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

  def set_embedding_weight_unk(self):
    self.embedding.weight.data[0] = torch.mean(self.embedding.weight.data[2:],0)


