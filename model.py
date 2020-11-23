import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, dropout, layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_size)
        self.hidden_size = hidden_size
        if dropout == 0:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers=layers)

    def forward(self, input_seq):
        embedding = self.embedding(input_seq)
        embedding = embedding.permute(1, 0, 2)
        if self.dropout is not None:
            embedding = self.dropout(embedding)
        output, hidden = self.rnn(embedding)
        return hidden.squeeze(0)


class Model(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, rnn_layers=1, dropout=0.2,
                 enc_dropout=0.2):
        super(Model, self).__init__()
        self.encoder = Encoder(input_size, embedding_size, hidden_size, enc_dropout, rnn_layers)
        self.encoder_layers = rnn_layers
        self.dropout = nn.Dropout(dropout)
        if rnn_layers > 1:
            self.compress = nn.Linear(hidden_size * rnn_layers, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_batch):
        hidden = self.encoder(input_batch)
        if self.encoder_layers > 1:
            concat = torch.cat([hidden[0], hidden[1]], dim=-1)
            hidden = self.compress(concat)
        hidden = self.dropout(hidden)
        output = self.out(hidden)
        return output
