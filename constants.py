import torch

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
FILENAME = 'data/airline_sentiment_analysis.csv'
PROCESSED_FILENAME = 'data/airline_sentiment_analysis_processed.csv'
# MAX_SENTENCE_LEN = 25
BIDIRECTIONAL = True
HIDDEN_SIZE = 128
N_LAYERS = 2
BATCH_SIZE = 32
LIN_DROPOUT = 0.2
EPOCHS = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
