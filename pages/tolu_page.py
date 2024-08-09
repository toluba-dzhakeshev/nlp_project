import streamlit as st
import joblib
import json
import time
import torch
from transformers import BertTokenizer, BertModel
from models.dependecies import predict_with_bert
from models.dependecies import (clean_text_transformer, tokenize_text_transformer, remove_stopwords_transformer,
                         lemmatize_tokens_transformer, join_tokens_transformer)
from models.dependecies import ConfigRNN, LSTMClassifier, vocab_to_int, predict_sentence

#TF-IDF
tfidf_model = joblib.load('./models/sentiment_model_pipeline.pkl')
with open('./models/label_mapping.json', 'r') as f:
    tfidf_label_mapping = json.load(f)
tfidf_index_to_label = {v: k for k, v in tfidf_label_mapping.items()}
    
#LSTM
lstm_config = joblib.load('./models/lstm_model_config.pkl')
seq_len = lstm_config.seq_len
lstm_model = LSTMClassifier(lstm_config)
lstm_model.load_state_dict(torch.load('./models/lstm_model_weights.pth', map_location=torch.device('cpu')))
lstm_model.eval()
lstm_label_mapping = {0: 'negative', 1: 'positive'}

#BERT
bert_model = BertModel.from_pretrained('tolubai/tolubai-bert-test')
bert_tokenizer = BertTokenizer.from_pretrained('tolubai/tolubai-bert-test')
logistic_model = joblib.load('./models/logistic_regression_model.pkl')
bert_label_mapping = {0: 'negative', 1: 'positive'}

#Streamlit code
st.title('Sentiment Analysis')

# Model selection
model_choice = st.selectbox(
    "Choose a model:",
    ("TF-IDF", "LSTM", "BERT")
)

user_input = st.text_area("Enter your text:")

if st.button('Predict Sentiment'):
    if user_input:
        start_time = time.time()

        if model_choice == "TF-IDF":
            # TF-IDF model prediction
            prediction = tfidf_model.predict([user_input])[0]
            predicted_label = tfidf_index_to_label[prediction]
        
        elif model_choice == "LSTM":
            # LSTM model prediction
            predicted_label = predict_sentence(user_input, lstm_model, seq_len, vocab_to_int)
        
        elif model_choice == "BERT":
            # BERT model prediction
            predicted_label = predict_with_bert(user_input, bert_model, bert_tokenizer, logistic_model)

        end_time = time.time()
        response_time = end_time - start_time

        st.write(f"Predicted Sentiment: **{predicted_label.capitalize()}**")
        st.write(f"Response Time: **{response_time:.4f} seconds**")
    else:
        st.write("Please enter text to analyze.")
        
st.write(10 * "---")
st.write("TF-IDF F1 Macro Score: 0.9080")
st.write("LSTM F1 Macro Score: 0.8760")
st.write("BERT F1 Macro Score: 0.8491")