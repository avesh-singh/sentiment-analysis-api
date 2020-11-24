import torch

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
FILENAME = 'data/airline_sentiment_analysis.csv'
MAX_SENTENCE_LEN = 15
EMBEDDING_SIZE = 200
HIDDEN_SIZE = 64
N_LAYERS = 1
BATCH_SIZE = 32
LIN_DROPOUT = 0.2
ENC_DROPOUT = 0.1
EPOCHS = 4
VOCAB_SIZE = 4000
EPS = 0.01
device = "cuda" if torch.cuda.is_available() else "cpu"
