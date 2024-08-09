import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pymorphy3
import re
import string

import numpy as np
import json
from dataclasses import dataclass
from torch import nn

import torch

#TF-IDF
morph = pymorphy3.MorphAnalyzer()
russian_stop_words = set(stopwords.words('russian'))

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r' ', text)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', " ", text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = remove_emojis(text)
    text = re.sub(r'\s+', ' ', text).strip()  # extra white-space
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'<.*?>', ' ', text)  # html tags
    return text

def tokenize_text(text):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(text)

def remove_stopwords(tokens):
    return [word for word in tokens if word not in russian_stop_words]

def lemmatize_tokens(tokens):
    return [morph.parse(token)[0].normal_form for token in tokens]

def join_tokens(tokens):
    return ' '.join(tokens)

# Wrapping transformers to handle lists and pandas Series
def clean_text_transformer(X):
    if isinstance(X, pd.Series):
        return X.apply(clean_text)
    elif isinstance(X, list):
        return [clean_text(text) for text in X]
    else:
        return clean_text(X)

def tokenize_text_transformer(X):
    if isinstance(X, pd.Series):
        return X.apply(tokenize_text)
    elif isinstance(X, list):
        return [tokenize_text(text) for text in X]
    else:
        return tokenize_text(X)

def remove_stopwords_transformer(X):
    if isinstance(X, pd.Series):
        return X.apply(remove_stopwords)
    elif isinstance(X, list):
        return [remove_stopwords(tokens) for tokens in X]
    else:
        return remove_stopwords(X)

def lemmatize_tokens_transformer(X):
    if isinstance(X, pd.Series):
        return X.apply(lemmatize_tokens)
    elif isinstance(X, list):
        return [lemmatize_tokens(tokens) for tokens in X]
    else:
        return lemmatize_tokens(X)

def join_tokens_transformer(X):
    if isinstance(X, pd.Series):
        return X.apply(join_tokens)
    elif isinstance(X, list):
        return [join_tokens(tokens) for tokens in X]
    else:
        return join_tokens(X)
    
#LSTM
with open('./models/vocab_to_int.json', 'r') as f:
    vocab_to_int = json.load(f)
    
russian_stop_words = pd.read_csv('./pages/stopwords-ru.txt', header=None)
russian_stop_words = set(russian_stop_words[0])

from nltk.corpus import stopwords
english_stop_words = set(stopwords.words('english'))

stop_words = english_stop_words.union(russian_stop_words)

@dataclass
class ConfigRNN:
    vocab_size: int
    device: str
    n_layers: int
    embedding_dim: int
    hidden_size: int
    seq_len: int
    bidirectional: bool
    
class LSTMClassifier(nn.Module):
    def __init__(self, rnn_conf=ConfigRNN) -> None:
        super().__init__()

        self.embedding_dim = rnn_conf.embedding_dim
        self.hidden_size = rnn_conf.hidden_size
        self.bidirectional = rnn_conf.bidirectional
        self.n_layers = rnn_conf.n_layers

        self.embedding = nn.Embedding(rnn_conf.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            bidirectional=self.bidirectional,
            batch_first=True,
            num_layers=self.n_layers,
        )
        self.bidirect_factor = 2 if self.bidirectional else 1
        self.clf = nn.Sequential(
            nn.Linear(self.hidden_size * self.bidirect_factor, 32),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(32, 1)
        )

    def model_description(self):
        direction = "bidirect" if self.bidirectional else "onedirect"
        return f"lstm_{direction}_{self.n_layers}"

    def forward(self, x: torch.Tensor):
        embeddings = self.embedding(x)
        out, _ = self.lstm(embeddings)
        out = out[:, -1, :]
        out = self.clf(out)
        return out
    
def data_preprocessing(text: str) -> str:
    text = text.lower()
    text = re.sub("<.*?>", "", text)  # Remove HTML tags
    text = "".join([c for c in text if c not in string.punctuation])
    splitted_text = [word for word in text.split() if word not in stop_words]
    text = " ".join(splitted_text)
    return text

def padding(review_int: list, seq_len: int) -> np.array:
    features = np.zeros((len(review_int), seq_len), dtype = int)
    for i, review in enumerate(review_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[: seq_len]
        features[i, :] = np.array(new)
    return features

def preprocess_single_string(input_string: str, seq_len: int, vocab_to_int: dict) -> torch.Tensor:
    preprocessed_string = data_preprocessing(input_string)
    result_list = [vocab_to_int.get(word, 0) for word in preprocessed_string.split()]
    result_padded = padding([result_list], seq_len)[0]
    return torch.tensor(result_padded)

def predict_sentence(text: str, model: nn.Module, seq_len: int, vocab_to_int: dict, device: str = 'cpu') -> tuple:
    input_tensor = preprocess_single_string(text, seq_len, vocab_to_int).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()  
    
    predicted_label = "positive" if probability >= 0.5 else "negative"
    return predicted_label

#BERT
def predict_with_bert(text: str, bert_model, bert_tokenizer, logistic_model):
    encoded_input = bert_tokenizer(text, return_tensors='pt', max_length=64, truncation=True, padding='max_length')
    with torch.no_grad():
        model_output = bert_model(**encoded_input)
    
    # Use the CLS token representation as input to logistic regression
    cls_embedding = model_output.last_hidden_state[:, 0, :].numpy()
    
    # Get the prediction from logistic regression
    logistic_prediction = logistic_model.predict(cls_embedding)[0]
    print(f"Logistic Model Output: {logistic_prediction}")  # Debugging: Print the output label
    
    return logistic_prediction