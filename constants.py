import torch

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
FILENAME = 'data/airline_sentiment_analysis.csv'
MAX_SENTENCE_LEN = 25
BIDIRECTIONAL = True
HIDDEN_SIZE = 128
N_LAYERS = 1
BATCH_SIZE = 32
DROPOUT = 0.2
EPOCHS = 3
device = "cuda" if torch.cuda.is_available() else "cpu"
