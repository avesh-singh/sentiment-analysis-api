import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, dropout=0.2):
        super(Model, self).__init__()
        self.embedding = nn.EmbeddingBag(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.lin1 = nn.Linear(embedding_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_size)

    def forward(self, input_batch):
        embedding = self.embedding(input_batch)
        embedding = self.dropout(embedding)
        output = self.lin1(embedding)
        output = self.lin2(output)
        return output
