import torch

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
FILENAME = 'data/airline_sentiment_analysis.csv'
MAX_SENTENCE_LEN = 20
EMBEDDING_SIZE = 200
HIDDEN_SIZE = 32
N_LAYERS = 1
BIDIRECTIONAL = True
BATCH_SIZE = 32
LIN_DROPOUT = 0.2
ENC_DROPOUT = 0.1
EPOCHS = 10
VOCAB_SIZE = 5000
EPS = 0.01
device = "cuda" if torch.cuda.is_available() else "cpu"
