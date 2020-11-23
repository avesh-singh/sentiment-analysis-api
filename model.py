import torch.nn as nn
import torch
from transformers import BertModel


class Model(nn.Module):
    def __init__(self, hidden_size, output_size, rnn_layers=1, dropout=0.1, bidirectional=False):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        embedding_dim = self.bert.config.to_dict()['hidden_size']
        if rnn_layers > 1:
            self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=rnn_layers, bidirectional=bidirectional,
                              batch_first=True, dropout=0 if bidirectional else dropout)
        else:
            self.gru = nn.GRU(embedding_dim, hidden_size, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)

    def forward(self, input_batch):
        with torch.no_grad():
            embedding = self.bert(input_batch)[0]
        _, hidden = self.gru(embedding)
        if self.gru.bidirectional:
            hidden = self.dropout(torch.cat((hidden[0], hidden[1]), dim=1))
        else:
            hidden = self.dropout(hidden[0])
        output = self.out(hidden)
        return output
