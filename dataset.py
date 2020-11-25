from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizer
from nltk.tokenize import TweetTokenizer


class AirlineTweetsDataset(Dataset):
    def __init__(self, tweets, sentiments, tokenizer, max_len):
        self.texts = tweets
        self.targets = sentiments
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        target = self.targets[item]

        encoding = self.tokenizer(
            text, self.max_len
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

    def get_data_loader(self, batch_size):
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=8
        )


class Tokenizer(object):
    def __init__(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.tweet_tokenizer = TweetTokenizer(strip_handles=True)

    def __call__(self, text, max_len):
        tokens = self.tweet_tokenizer.tokenize(text)
        return self.bert_tokenizer(
            tokens,
            is_split_into_words=True,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
