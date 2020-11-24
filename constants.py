import torch

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
FILENAME = 'data/airline_sentiment_analysis.csv'
PROCESSED_FILENAME = 'data/airline_sentiment_analysis_processed.csv'
MAX_SENTENCE_LEN = 15
EMBEDDING_SIZE = 200
HIDDEN_SIZE = 32
N_LAYERS = 1
BATCH_SIZE = 32
LIN_DROPOUT = 0.2
EPOCHS = 8
VOCAB_SIZE = 5000
device = "cuda" if torch.cuda.is_available() else "cpu"
