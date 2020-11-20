import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, dropout=0.2):
        super(Model, self).__init__()
        self.embedding = nn.EmbeddingBag(num_embeddings=input_size, embedding_dim=embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.lin1 = nn.Linear(embedding_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_size)

    def forward(self, input_batch):
        embedding = self.dropout(self.embedding(input_batch))
        output = self.lin1(embedding)
        output = self.lin2(output)
        return output
