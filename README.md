# Sentiment Analysis model comparison

## Problem Statement:

Create a RESTful API which accepts English text as input and return the associated sentiment as JSON.

## Dataset:

A labelled dataset of tweets by related to various airlines has been provided. Dataset is very imbalanced with only
20% tweets with positive sentiment. It should be balanced to get meaningful results. Both oversampling the positive
reviews and under sampling the negative reviews was tried. Final dataset used for training uses over sampling
strategy. Dataset was divided into 70%:15%:15% chunks(train: validation: test)

#Preprocessing:
Since this dataset had tweets, emojis(❤️, 👍) and emoticons(:/ , :( ) were replaced with corresponding english
description (‘red heart’, ‘thumbs up’, ‘annoyed’, ‘sad’) using demoji and emot packages. After that, NLTK’s
TweetTokenizer was used to clean up data. Conversion of processed sentences to numerical tokens was done
seperately by __Scikit-learn's__ _CountVectorizer_, __Spacy’s__ _Tokenizer_, __Keras’__ _Tokenizer_ and __HuggingFace Transformers’__ _BertTokenizer_
during model fitting process. Final model is __bert-base-cased__ model and it requires that input tokens are tokenized using
WordPiece algorithm. Hence, BertTokenizer is used in final model.

# Training:
Summary of models trained with their accuracy is as below:

Model | Training Accuracy % | Test Accuracy % | Test Precision | Test Recall |
-| - | - | - | - |
Naïve Bayes | 93.45 | 73.90 | 0.4198 | 0.7337 |
Linear NN | 94.64 | 91.09 | 0.7867 | 0.7734 |
RNN | 94.23 | 89.99 | 0.7812 | 0.7082 |
BERT | 98.98 | 93.13 | 0.8556 | 0.8263 |

__Naïve Bayes__: NLTK’s GaussianNB model, with count vectorized data

__Linear NN__: Double Layer Feed forward Neural network, first layer being word embeddings with 0.2 dropout

__RNN__: Encoder is Single layer Bidirectional Recurrent Neural Network with 0.1 dropout; Decoder is single layer Feed
forward NN with 0.2 dropout

__BERT__: Bert-base model pretrained on BookCorpus and English Wikipedia is used as trained base with a single layer
Bidirectional Recurrent Neural Network as classification head. Bert-base has 12 transformer encoder layers of 768
units each with 12 attention heads.
