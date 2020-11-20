from torchtext.data import TabularDataset, Field, LabelField, Example
from torchtext.data.iterator import BucketIterator, Iterator
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader, BatchSampler
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import spacy
import re
import torch
from warnings import filterwarnings
from string import punctuation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import random
import numpy as np

filterwarnings('ignore', '.* class will be retired')

spacy_en = spacy.load('en')
device = "cuda" if torch.cuda.is_available() else "cpu"
counts = {'only_words': set(), 'others': set()}
hashtag = re.compile(r'(?:#|@)(\w+\s?)')
token_pattern = re.compile(r'[a-zA-Z]+')
stopwords = spacy_en.Defaults.stop_words.union(set(punctuation))
tokenizer = Tokenizer(oov_token='<unk>', filters='', num_words=5000)


def en_tokenize(sentence):
    if re.search(hashtag, sentence) is not None:
        sentence = re.sub(hashtag, r'\1', sentence)
    tokens = [tok.text for tok in spacy_en.tokenizer(sentence)]
    tokens = [tok for tok in tokens if tok not in stopwords]
    return " ".join(tokens)


def prepare_data(batch_size):
    field = Field(tokenize=en_tokenize,
                  lower=True)
    label_field = LabelField()
    dataset = TabularDataset('data/airline_sentiment_analysis.csv', 'csv',
                             [('id', None), ('airline_sentiment', label_field), ('text', field)], skip_header=True)
    train, valid, test = dataset.split([0.8, 0.1, 0.1])
    field.build_vocab(train, min_freq=2)
    label_field.build_vocab(train)
    print(len(train), len(valid), len(test))
    train_iter, valid_iter, test_iter = Iterator.splits(
        (train, valid, test),
        batch_size=batch_size,
        device=device,
        # sort_key=lambda x: len(x.split()),
        # sort=True
    )
    # valid_iter = Iterator.splits(
    #     valid,
    #     batch_size=batch_size,
    #     device=device,
    #     sort_key=lambda x: len(x.split()),
    #     sort=True
    # )
    # test_iter = Iterator.splits(
    #     test,
    #     batch_size=batch_size,
    #     device=device,
    #     sort_key=lambda x: len(x.split()),
    #     sort=True
    # )
    return train_iter, valid_iter, test_iter, field, label_field


def convert_to_tensor(lst):
    return torch.tensor(lst, dtype=torch.long, device=device)


def convert_to_seqs(text):
    sequences = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(sequences, maxlen=15, truncating='post')
    return convert_to_tensor(padded)


def balance_classes(batch):
    # text, label = batch[0], batch[1]
    text = [t[0] for t in batch]
    label = [t[1] for t in batch]
    # print(text)
    desired_count = len(batch) // 2
    label_counts = Counter([t.item() for t in label])
    # print(label_counts)
    class_to_oversample = [key for key, value in label_counts.items() if value < desired_count]
    if len(class_to_oversample) == 0:
        print("no class is outnumbered", label_counts)
    else:
        class_to_oversample = class_to_oversample[0]
        indices_to_copy = [i for i, lbl in enumerate(label) if lbl == class_to_oversample]
        copy_instances = random.choices(indices_to_copy, k=desired_count - label_counts[class_to_oversample])
        text.extend([text[c] for c in copy_instances])
        label.extend([label[c] for c in copy_instances])
    return torch.stack(text), torch.stack(label)


def create_dataloader(text, label, batch_size, train=False):
    dataset = TensorDataset(convert_to_seqs(text), convert_to_tensor(label))
    if train:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=balance_classes, drop_last=True)
    # loader = DataLoader(dataset, batch_sampler=BatchSampler(sampler, batch_size, drop_last=False))
    return loader


def prepare_data_keras(batch_size):
    df = pd.read_csv('data/airline_sentiment_analysis.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df['text'] = df['text'].apply(en_tokenize)
    df['airline_sentiment'] = df['airline_sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    train_text, temp_text, train_label, temp_label = train_test_split(df['text'].values, df['airline_sentiment'].values,
                                                                      test_size=0.3, stratify=df['airline_sentiment'].values)
    valid_text, test_text, valid_label, test_label = train_test_split(temp_text, temp_label, test_size=0.5)
    tokenizer.fit_on_texts(train_text)
    print(len(tokenizer.word_index))
    print(f"{compute_class_weight('balanced', classes=[0, 1], y=train_label)}\n"
          f"{compute_class_weight('balanced', classes=[0, 1], y=valid_label)}\n"
          f"{compute_class_weight('balanced', classes=[0, 1], y=test_label)}")
    train_loader = create_dataloader(train_text, train_label, batch_size, True)
    valid_loader = create_dataloader(valid_text, valid_label, batch_size)
    test_loader = create_dataloader(test_text, test_label, batch_size)
    return train_loader, valid_loader, test_loader, len(tokenizer.word_index)
