import torch
import torch.nn as nn
from torchcrf import CRF


class Embedding_layer(nn.Module):
    def __init__(self,
                 word_input_dim,
                 word_embedding_dim,
                 char_input_dim,
                 char_embedding_dim,
                 char_cnn_filter_num,
                 char_cnn_kernel_size,
                 word_pad_idx,
                 char_pad_idx,

                 char_emb_dropout,
                 word_emb_dropout,
                 cnn_dropout,

                 use_char = True):
        """
        :param word_input_dim: number of words in vocab, each word -> vector
        :param word_embedding_dim: dim of word embedding vector
        :param char_input_dim: number of characters in vocab
        :param char_embedding_dim: dim of char embedding vector
        :param char_cnn_filter_num:
        :param char_cnn_kernel_size:
        :param word_pad_idx:
        :param char_pad_idx:
        :param char_emb_dropout:
        :param word_emb_dropout:
        :param cnn_dropout:
        """
        super().__init__()
        self.word_pad_idx = word_pad_idx
        self.char_pad_idx = char_pad_idx
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.use_char = use_char

        # LAYER 1A: Word Embedding
        self.word_embedding = nn.Embedding(
            num_embeddings=word_input_dim,
            embedding_dim=word_embedding_dim,
            padding_idx=word_pad_idx  # the entries at padding_idx do not contribute to the gradient
        )
        self.word_emb_dropout = nn.Dropout(word_emb_dropout)

        # LAYER 1B: Char Embedding-CNN
        self.char_embedding = nn.Embedding(
            num_embeddings=char_input_dim,
            embedding_dim=char_embedding_dim,
            padding_idx=char_pad_idx  # the entries at padding_idx do not contribute to the gradient
        )
        self.char_emb_dropout = nn.Dropout(char_emb_dropout)
        self.char_cnn = nn.Conv1d(
            in_channels=char_embedding_dim,
            out_channels=char_embedding_dim * char_cnn_filter_num,
            kernel_size=char_cnn_kernel_size,
            # groups=char_embedding_dim  # different 1d conv for each embedding dim
        )
        self.cnn_dropout = nn.Dropout(cnn_dropout)

    def forward(self, words, chars):
        """shape of input
        # words = [sentence length, batch size]
        # chars = [batch size, sentence length, word length)
        # tags = [sentence length, batch size]
        """

        # word_emb = [sentence length, batch size, word emb dim]
        word_emb = self.word_emb_dropout(self.word_embedding(words))
        # print("word_emb shape: ", word_emb.shape)

        word_features = word_emb

        if self.use_char:
            # char_emb = [batch size, sentence length, word length, char emb dim]
            char_emb = self.char_emb_dropout(self.char_embedding(chars))
            # print("char_emb shape: ",char_emb.shape)

            batch_size, sent_len, word_len, char_emb_dim = char_emb.shape
            char_emb_cnn = torch.zeros(batch_size, sent_len, self.char_cnn.out_channels).to('cuda')
            for word_i in range(sent_len):
                # Embedding of word belong with character
                # char_emb_word_i = [batch size, word length, char emb dim]
                char_emb_word_i = char_emb[:, word_i, :, :]
                # input of Conv1d has shape [N, C_in, L_in]
                # -> permute:  char_emb_word_i = [batch size, char emb dim, word length]
                char_emb_word_i = char_emb_word_i.permute(0, 2, 1)
                # print("char_emb_word_i shape: ",char_emb_word_i.shape)

                # char_emb_word_i_out = [batch size, out channels, word length - kernel size + 1]
                char_emb_word_i_out = self.char_cnn(char_emb_word_i)
                # print("char_emb_word_i_out shape: ", char_emb_word_i_out.shape)

                # Max pooling from: [batch size, out channels, ...] --> [batch size, out channels]
                # Character-level representation : one word (many characters) -> one vector 125-dim
                char_emb_cnn[:, word_i, :], _ = torch.max(char_emb_word_i_out, dim=2)

            # print("char_emb_cnn shape: ",char_emb_cnn.shape)
            # char_emb_cnn = [batch size, sentence length, out channels]
            char_emb_cnn = self.cnn_dropout(char_emb_cnn)
            # because word_emb has shape [sentence length, batch size, word emb dim]
            # -> permute char_emb_cnn to [sentence length, batch size, out channels]
            char_emb_cnn = char_emb_cnn.permute(1, 0, 2)
            # Concat word_emb and char_emb_cnn to get word_features
            # Shape [sentence length, batch size, word emb dim + out channels]
            # print(char_emb_cnn.is_cuda, word_emb.is_cuda)
            word_features = torch.cat((word_emb, char_emb_cnn), dim=2)

        return word_features

    def init_embeddings(self):
        # initialize embedding for padding as zero
        self.word_embedding.weight.data[self.word_pad_idx] = torch.zeros(self.word_embedding_dim)
        if self.use_char:
            self.char_embedding.weight.data[self.char_pad_idx] = torch.zeros(self.char_embedding_dim)


class CRF_layer(nn.Module):
    def __init__(self, input_dim, output_dim, fc_dropout, tag_pad_idx):
        super().__init__()
        self.tag_pad_idx = tag_pad_idx
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear_dropout = nn.Dropout(fc_dropout)
        self.crf = CRF(num_tags=output_dim)

    def forward(self, lstm_features, tags):
        fc_out = self.linear_dropout(self.linear(lstm_features))

        # For training
        if tags is not None:
            mask = tags != self.tag_pad_idx
            crf_out = self.crf.decode(fc_out, mask=mask)
            crf_loss = -self.crf(fc_out, tags=tags, mask=mask)

        # For testing
        else:
            crf_out = self.crf.decode(fc_out)
            crf_loss = None

        return crf_out, crf_loss

    def init_crf_transitions(self, tag_list, imp_value=-1e4):
        """
        :param tag_list: ['<pad>','O','B-LOCATION','I-LOCATION','B-PATIENT_ID',...]
        :param imp_value: value that we assign for impossible transition, ex: b-location -> i-patient_id
        """
        num_tags = len(tag_list)
        for i in range(num_tags):
            tag_name = tag_list[i]
            # I and <pad> impossible as a start tag
            if tag_name[0] == "I" or tag_name == "<pad>":
                nn.init.constant_(self.crf.start_transitions[i], imp_value)
            # No impossible as an end

        prefix_dict = {}
        for tag_position in ("B", "I", "O"):
            prefix_dict[tag_position] = [i for i, tag in enumerate(tag_list) if tag[0] == tag_position]
        # prefix_dict =
        # {'B': [2, 4, 5, 9, 10, 11, 12, 13, 14, 15],
        #  'I': [3, 6, 7, 8, 16, 17, 18, 19, 20],
        #  'O': [1]}

        # init impossible transitions between positions
        impossible_transitions_position = {"O": "I"}
        for prefix_1, prefix_2 in impossible_transitions_position.items():
            for i in prefix_dict[prefix_1]:
                for j in prefix_dict[prefix_2]:
                    nn.init.constant_(self.crf.transitions[i, j], imp_value)

        # init impossible B and I transition to different entity types
        impossible_transitions_tags = {"B": "I", "I": "I"}
        for prefix_1, prefix_2 in impossible_transitions_tags.items():
            for i in prefix_dict[prefix_1]:
                for j in prefix_dict[prefix_2]:
                    if tag_list[i].split("-")[1] != tag_list[j].split("-")[1]:
                        nn.init.constant_(self.crf.transitions[i, j], imp_value)


class lstm_crf(nn.Module):
    def __init__(self,
                 word_input_dim,
                 word_embedding_dim,
                 char_input_dim,
                 char_embedding_dim,
                 char_cnn_filter_num,
                 char_cnn_kernel_size,
                 lstm_hidden_dim,
                 output_dim,
                 lstm_layers,

                 char_emb_dropout,
                 word_emb_dropout,
                 cnn_dropout,
                 lstm_dropout,
                 fc_dropout,

                 word_pad_idx,
                 char_pad_idx,
                 tag_pad_idx,

                 use_char = True):
        super().__init__()
        self.word_pad_idx = word_pad_idx
        self.tag_pad_idx = tag_pad_idx
        self.char_pad_idx = char_pad_idx
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim

        self.embedding_layer = Embedding_layer(word_input_dim=word_input_dim,
                                               word_embedding_dim=word_embedding_dim,
                                               char_input_dim=char_input_dim,
                                               char_embedding_dim=char_embedding_dim,
                                               char_cnn_filter_num=char_cnn_filter_num,
                                               char_cnn_kernel_size=char_cnn_kernel_size,
                                               word_pad_idx=word_pad_idx,
                                               char_pad_idx=char_pad_idx,

                                               char_emb_dropout=char_emb_dropout,
                                               word_emb_dropout=word_emb_dropout,
                                               cnn_dropout=cnn_dropout,
                                               use_char=use_char,
                                               )

        self.lstm = nn.LSTM(
            input_size=word_embedding_dim + use_char*(char_embedding_dim * char_cnn_filter_num),
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0
        )

        self.crf_layer = CRF_layer(input_dim=lstm_hidden_dim * lstm_layers,
                                   output_dim=output_dim,
                                   fc_dropout=fc_dropout,
                                   tag_pad_idx=tag_pad_idx)

        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)

    def forward(self, words, chars, tags=None):
        word_features = self.embedding_layer(words, chars)
        lstm_features, _ = self.lstm(word_features)
        crf_out, crf_loss = self.crf_layer(lstm_features, tags)

        return crf_out, crf_loss

    def init_crf_transitions(self, tag_list):
        self.crf_layer.init_crf_transitions(tag_list, imp_value=-1e4)

    def init_embeddings(self):
        self.embedding_layer.init_embeddings()

    def save_state(self, path):
        torch.save(self.state_dict(), path)

    def load_state(self, path):
        self.load_state_dict(torch.load(path))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)