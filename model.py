import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, dropout, layers, bidirectional):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_size)
        self.hidden_size = hidden_size
        if dropout == 0:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers=layers, bidirectional=bidirectional)

    def forward(self, input_seq):
        embedding = self.embedding(input_seq)
        embedding = embedding.permute(1, 0, 2)
        if self.dropout is not None:
            embedding = self.dropout(embedding)
        output, hidden = self.rnn(embedding)
        return hidden.squeeze(0)


class Model(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, rnn_layers=1, dropout=0.2,
                 enc_dropout=0.2, bidirectional=True):
        super(Model, self).__init__()
        self.encoder = Encoder(input_size, embedding_size, hidden_size, enc_dropout, rnn_layers, bidirectional)
        self.encoder_layers = rnn_layers
        self.dropout = nn.Dropout(dropout)
        self.directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        if rnn_layers > 1 or bidirectional:
            self.compress = nn.Linear(hidden_size * rnn_layers * self.directions, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_batch):
        hidden = self.encoder(input_batch)
        if self.encoder_layers > 1 or self.encoder.bidirectional:
            hidden_layers = []
            for i in range(self.encoder_layers * self.directions):
                hidden_layers.append(hidden[i])
            concat = torch.cat(hidden_layers, dim=-1)
            hidden = self.compress(concat)
        hidden = self.dropout(hidden)
        output = self.out(hidden)
        return output
