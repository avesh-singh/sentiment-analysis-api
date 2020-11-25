import re
import random
import demoji
import emot

random.seed(6)
hashtag = re.compile(r'(?:#|@)(\w+\s?)')
links = re.compile(r'https?://[^ ]+')
mean_exp = re.compile(r'([a-zA-Z]+(?:, )?)')


def extract_emotion(means):
    meanings = means.split(',')
    return random.choice(meanings)


def clean_sentence(sentence):
    if re.search(hashtag, sentence) is not None:
        sentence = re.sub(hashtag, r'\1', sentence)
    sentence = re.sub(links, 'URL', sentence)
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
