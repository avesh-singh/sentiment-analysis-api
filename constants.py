import torch

torch.manual_seed(1)
FILENAME = 'data/airline_sentiment_analysis.csv'
MAX_SENTENCE_LEN = 15
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 64
N_LAYERS = 1
BATCH_SIZE = 32
DROPOUT = 0.1
EPOCHS = 8
VOCAB_SIZE = 5000
EPS = 0.01
device = "cuda" if torch.cuda.is_available() else "cpu"
