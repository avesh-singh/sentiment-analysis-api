from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
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
from constants import VOCAB_SIZE, device, BATCH_SIZE, FILENAME, MAX_SENTENCE_LEN
import pickle
import numpy as np
from utils import clean_sentence

# filterwarnings('ignore', '.* class will be retired')

spacy_en = spacy.load('en')
np.random.seed(2)
counts = {'only_words': set(), 'others': set()}
hashtag = re.compile(r'(?:#|@)(\w+\s?)')
token_pattern = re.compile(r'[a-zA-Z]+')
stopwords = spacy_en.Defaults.stop_words.union(set(punctuation))
tokenizer = Tokenizer(oov_token='<unk>', filters='', num_words=VOCAB_SIZE)


def en_tokenize(sentence):
    """
    removes stopwords and converts tweet hastags/mentions to normal words
    :param sentence: str, raw sentence from dataset
    :return: clean sentence
    """
    tokens = [tok.text for tok in spacy_en.tokenizer(clean_sentence(sentence.lower()))]
    tokens = [tok for tok in tokens if tok not in stopwords]
    return " ".join(tokens)


# def prepare_data(batch_size):
#     field = Field(tokenize=en_tokenize,
#                   lower=True)
#     label_field = LabelField()
#     dataset = TabularDataset('data/airline_sentiment_analysis.csv', 'csv',
#                              [('id', None), ('airline_sentiment', label_field), ('text', field)], skip_header=True)
#     train, valid, test = dataset.split([0.8, 0.1, 0.1])
#     field.build_vocab(train, min_freq=2)
#     label_field.build_vocab(train)
#     print(len(train), len(valid), len(test))
#     train_iter, valid_iter, test_iter = Iterator.splits(
#         (train, valid, test),
#         batch_size=batch_size,
#         device=device,
#     )
#     return train_iter, valid_iter, test_iter, field, label_field


def convert_to_tensor(lst):
    return torch.tensor(lst, dtype=torch.long, device=device)


def convert_to_seqs(tokenizer, text):
    """
    converts list of sentences to a torch.Tensor of padded sequences of 'MAX_SENTENCE_LEN' length based on tokenizer
    :param tokenizer: instance of keras_preprocessing.text.Tokenizer already fitted on data
    :param text: list of sentences to be tokenized
    :return: torch.Tensor of size len(text) x MAX_SENTENCE_LEN
    """
    text = list(map(en_tokenize, text))
    sequences = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(sequences, maxlen=MAX_SENTENCE_LEN, truncating='pre')
    return convert_to_tensor(padded)


def collate_batch(batch):
    text = [t[0] for t in batch]
    label = [t[1] for t in batch]
    return torch.stack(text), torch.stack(label)


def create_data_loader(text, label, train=False):
    dataset = TensorDataset(convert_to_seqs(tokenizer, text), convert_to_tensor(label))
    if train:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE, collate_fn=collate_batch, drop_last=True)
    return loader


def balance_data(text, labels):
    """
    This function balances the dataset by oversampling the minor class. Final response will have equal number of
    examples from all the classes
    :param text: array of texts
    :param labels: array of corresponding labels
    :return: tuple of oversampled texts and labels
    """
    data = np.array([text, labels]).T
    uniq, uniq_idx = np.unique(data[:, -1], return_inverse=True)
    uniq_cnt = np.bincount(uniq_idx)
    cnt = np.max(uniq_cnt)
    out = np.empty((cnt * len(uniq) - len(data), data.shape[1]), data.dtype)
    slices = np.concatenate(([0], np.cumsum(cnt - uniq_cnt)))
    for j in range(len(uniq)):
        indices = np.random.choice(np.where(uniq_idx == j)[0], cnt - uniq_cnt[j])
        out[slices[j]:slices[j + 1]] = data[indices]
    data = np.vstack((data, out))
    return data[:, 0], data[:, 1].astype(np.int32)


def prepare_data_keras():
    df = pd.read_csv(FILENAME)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df['airline_sentiment'] = df['airline_sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    train_text, temp_text, train_label, temp_label = train_test_split(df['text'].values, df['airline_sentiment'].values,
                                                                      test_size=0.3,
                                                                      stratify=df['airline_sentiment'].values,
                                                                      random_state=4)
    train_text, train_label = balance_data(train_text, train_label)
    valid_text, test_text, valid_label, test_label = train_test_split(temp_text, temp_label, test_size=0.5,
                                                                      stratify=temp_label)
    tokenizer.fit_on_texts(train_text)
    pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
    print(len(tokenizer.word_index))
    print(f"{compute_class_weight('balanced', classes=[0, 1], y=train_label)}\n"
          f"{compute_class_weight('balanced', classes=[0, 1], y=valid_label)}\n"
          f"{compute_class_weight('balanced', classes=[0, 1], y=test_label)}")
    train_loader = create_data_loader(train_text, train_label, True)
    valid_loader = create_data_loader(valid_text, valid_label)
    test_loader = create_data_loader(test_text, test_label)
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    prepare_data_keras()
