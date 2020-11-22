import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, dropout=0.2, layers=1):
        super(Model, self).__init__()
        self.embedding = nn.EmbeddingBag(num_embeddings=input_size, embedding_dim=embedding_size)
        if dropout == 0:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)
        self.layers = layers
        self.linear_layers = []
        if layers > 1:
            span = embedding_size - hidden_size
            compression = span // layers
            inp = embedding_size
            for i in range(layers):
                out = embedding_size - compression * (i + 1)
                nm = 'lin{}'.format(i)
                self.linear_layers.append(nm)
                setattr(self, nm, nn.Linear(inp, out))
                inp = out
        else:
            self.linear_layers.append('lin{}'.format(layers - 1))
            setattr(self, self.linear_layers[-1], nn.Linear(embedding_size, hidden_size))
        self.linear_layers.append('lin{}'.format(layers))
        setattr(self, self.linear_layers[-1], nn.Linear(hidden_size, output_size))

    def forward(self, input_batch):
        embedding = self.embedding(input_batch)
        if self.dropout is not None:
            embedding = self.dropout(embedding)
        output = embedding
        for layer in self.linear_layers:
            output = getattr(self, layer)(output)
        return output
