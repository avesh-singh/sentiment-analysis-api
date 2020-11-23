from torchtext import data
import re
import torch
from warnings import filterwarnings
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from constants import device, BATCH_SIZE, FILENAME, PROCESSED_FILENAME
import numpy as np
from transformers import BertTokenizer
import os

filterwarnings('ignore', '.* class will be retired')
np.random.seed(2)
counts = {'only_words': set(), 'others': set()}
hashtag = re.compile(r'(?:#|@)(\w+\s?)')
token_pattern = re.compile(r'[a-zA-Z]+')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_SENTENCE_LEN = tokenizer.max_model_input_sizes['bert-base-uncased']


def en_tokenize(sentence):
    """
    removes stopwords and converts tweet hastags/mentions to normal words
    :param sentence: str, raw sentence from dataset
    :return: clean sentence
    """
    if re.search(hashtag, sentence) is not None:
        sentence = re.sub(hashtag, r'\1', sentence)
    tokens = tokenizer.tokenize(sentence)
    # since bert model has maximum sentence length preset at 512, we are clipping the tokens to desired len in
    # addition to leaving space for special tokens
    tokens = tokens[:MAX_SENTENCE_LEN - 2]
    return tokens


def prepare_data():
    if not os.path.isfile(PROCESSED_FILENAME):
        df = pd.read_csv(FILENAME)
        df.drop(columns=['Unnamed: 0'], inplace=True)
        df['airline_sentiment'] = df['airline_sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
        df.to_csv(PROCESSED_FILENAME, index=False)
    TEXT = data.Field(
        use_vocab=False,
        batch_first=True,
        tokenize=en_tokenize,
        preprocessing=tokenizer.convert_tokens_to_ids,
        init_token=tokenizer.cls_token_id,
        eos_token=tokenizer.sep_token_id,
        pad_token=tokenizer.pad_token_id,
        unk_token=tokenizer.unk_token_id
    )
    LABEL = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
    dataset = data.TabularDataset(
        PROCESSED_FILENAME, 'csv', [('airline_sentiment', LABEL), ('text', TEXT)], skip_header=True
    )
    train, test = dataset.split(0.1, stratified=True, strata_field='airline_sentiment')
    valid, test = test.split(0.1)
    train_loader, valid_loader, test_loader = data.BucketIterator.splits(
        (train, valid, test),
        batch_size=BATCH_SIZE,
        device=device,
        sort=True,
        sort_key=lambda x: len(x.text)
    )
    class_weights = torch.tensor(
        compute_class_weight('balanced', classes=[0, 1], y=list(map(int, train.airline_sentiment))),
        device=device
    )
    return train_loader, valid_loader, test_loader, class_weights


if __name__ == '__main__':
    prepare_data()
