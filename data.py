from warnings import filterwarnings
import pandas as pd
from sklearn.model_selection import train_test_split
from constants import BATCH_SIZE, MAX_SENTENCE_LEN, FILENAME
import numpy as np
from collections import Counter
import random
from dataset import *
import emot
import demoji

filterwarnings('ignore', '.* class will be retired')
np.random.seed(2)
random.seed(2)


def extract_emotion(means):
    meanings = means.split(',')
    return random.choice(meanings)


def clean_sentence(sentence):
    reference = demoji.findall(sentence)
    # print(reference)
    emoticons = emot.emoticons(sentence)
    if isinstance(emoticons, list):
        emoticons = emoticons[0]
    # print(emoticons)
    if len(reference) > 0:
        for key, value in reference.items():
            sentence = sentence.replace(key, value+" ")
    if emoticons['flag']:
        for i in range(len(emoticons['value'])):
            # print(emoticons['value'][i])
            sentence = sentence.replace(emoticons['value'][i], extract_emotion(emoticons['mean'][i]))
    return sentence


def create_data_loader(df, tokenizer, max_len, batch_size):
    dataset = AirlineTweetsDataset(
        tweets=df.text.to_numpy(),
        sentiments=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return dataset.get_data_loader(batch_size)


def balance_data(df, strategy='under'):
    """
    This function balances the input dataframe in case of imbalance data
    :param df: pandas.DataFrame
    :param strategy: str, either 'over' for over-sampling or 'under' for under-sampling
    :return: pandas.DataFrame, dataframe with equal number of samples for both classes
    """
    func = None
    assert strategy in ['over', 'under'], "strategy should be either 'over' or 'under'"
    if strategy == 'over':
        func = max
    elif strategy == 'under':
        func = min
    labels = Counter(df['sentiment'])
    imp_class_count = func(labels.items(), key=lambda x: x[1])
    imp_class_examples = df[df['sentiment'] == imp_class_count[0]]
    sampled_examples = df[df['sentiment'] != imp_class_count[0]].sample(n=imp_class_count[1], replace=True,
                                                                        random_state=5)
    final = pd.concat([imp_class_examples, sampled_examples], axis='rows')\
        .sample(frac=1, replace=True, random_state=5).reset_index()
    return final[['text', 'sentiment']]


def load_data():
    df = pd.read_csv(FILENAME)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['sentiment'] = df['airline_sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    df['text'] = df['text'].apply(clean_sentence)
    df_train, df_valid = train_test_split(df, test_size=0.3, random_state=5)
    df_valid, df_test = train_test_split(df_valid, test_size=0.5, random_state=5)
    df_train.reset_index(inplace=True)
    df_train = balance_data(df_train, 'over')
    tokenizer = Tokenizer()
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_SENTENCE_LEN, BATCH_SIZE)
    valid_data_loader = create_data_loader(df_valid, tokenizer, MAX_SENTENCE_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_SENTENCE_LEN, BATCH_SIZE)
    return train_data_loader, valid_data_loader, test_data_loader, len(df_train), len(df_valid), len(df_test)


if __name__ == '__main__':
    train_loader, valid_loader, test_loader, train_len, valid_len, test_len = load_data()
