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
from collections import Counter
import random
import demoji

filterwarnings('ignore', '.* class will be retired')
np.random.seed(2)
random.seed(2)
counts = {'only_words': set(), 'others': set()}
hashtag = re.compile(r'(?:#|@)(\w+\s?)')
# token_pattern = re.compile(r'[a-zA-Z]+')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_SENTENCE_LEN = tokenizer.max_model_input_sizes['bert-base-uncased']


def en_tokenize(sentence, tknzr=tokenizer):
    """
    removes stopwords and converts tweet hastags/mentions to normal words
    :param tknzr: transformers.tokenization_utils_base.PreTrainedTokenizerBase tokenizer instance
    :param sentence: str, raw sentence from dataset
    :return: clean sentence
    """
    if re.search(hashtag, sentence) is not None:
        sentence = re.sub(hashtag, r'\1', sentence)
    reference = demoji.findall(sentence)
    if len(reference) > 0:
        for key, value in reference.items():
            sentence = sentence.replace(key, value+" ")
        # print(reference)
    tokens = tknzr.tokenize(sentence)
    # since bert model has maximum sentence length preset at 512, we are clipping the tokens to desired len in
    # addition to leaving space for special tokens
    tokens = tokens[:MAX_SENTENCE_LEN - 2]
    return tokens


def balance_data(dataset, strategy='under'):
    """
    this class balances the dataset for binary classification problem
    :param strategy: str, either 'over' or 'under' for 'oversampling' and 'undersampling'
    :param dataset: torchtext.data.dataset.TabularDataset
    :return: does not return anything, dataset is modified inplace
    """
    func = None
    assert strategy in ['over', 'under'], "strategy should be either 'over' or 'under'"
    if strategy == 'over':
        func = max
    elif strategy == 'under':
        func = min
    labels = Counter(list(map(lambda x: x.sentiment, dataset.examples)))
    imp_class_count = func(labels.items(), key=lambda x: x[1])
    imp_class_examples = list(filter(lambda x: x.sentiment == imp_class_count[0], dataset.examples))
    sampled_examples = random.choices(list(filter(lambda x: x.sentiment != imp_class_count[0], dataset.examples)),
                                      k=imp_class_count[1])
    dataset.examples = imp_class_examples + sampled_examples


def prepare_data():
    if not os.path.isfile(PROCESSED_FILENAME):
        df = pd.read_csv(FILENAME)
        df.drop(columns=['Unnamed: 0'], inplace=True)
        df['sentiment'] = df['airline_sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
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
        PROCESSED_FILENAME, 'csv', [(None, None), ('text', TEXT), ('sentiment', LABEL)], skip_header=True
    )
    valid, train = dataset.split(0.15, stratified=True, strata_field='sentiment')
    balance_data(train)
    print("{} training examples\n{} validation examples".format(len(train), len(valid)))
    train_loader, valid_loader = data.BucketIterator.splits(
        (train, valid),
        batch_size=BATCH_SIZE,
        device=device,
        sort=True,
        sort_key=lambda x: len(x.text)
    )
    class_weights = torch.tensor(
        compute_class_weight('balanced', classes=[0, 1], y=list(map(int, train.sentiment))),
        device=device
    )
    return train_loader, valid_loader, None, class_weights


if __name__ == '__main__':
    prepare_data()
