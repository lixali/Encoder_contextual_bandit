<<<<<<< HEAD
from transformers import BertModel, BertTokenizer
import torch


def getBertEmbedding(sentence):
# Load pre-trained BERT model and tokenizer
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Define input text
    input_text = sentence
    input_text2 = "hello world!"

    # Tokenize input text
    print(tokenizer.encode(input_text2, add_special_tokens=True))
    input_ids = torch.tensor([tokenizer.encode(input_text2, add_special_tokens=True)])

    # Get BERT embeddings for the input text
    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_states = outputs[0]

    # Print BERT embeddings
    #print(last_hidden_states.size())



'''
forward(X):
    bert_result = []
    i=0
    while i<len(X):
        lenght = 0
        current_sentence=""
        while(length<bert_maximum):
            current_sentence+=X[i]
            i+=1
        bert_result.append(self.bert(current_sentence))

    rnn_out = self.rnn(bert_result)
    ann_out = self.ann(rnn_out)

'''
=======
from transformers import BertModel, BertTokenizer
import torch


def getBertEmbedding(sentence):
# Load pre-trained BERT model and tokenizer
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Define input text
    input_text = sentence
    input_text2 = "hello world!"

    # Tokenize input text
    print(tokenizer.encode(input_text2, add_special_tokens=True))
    input_ids = torch.tensor([tokenizer.encode(input_text2, add_special_tokens=True)])

    # Get BERT embeddings for the input text
    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_states = outputs[0]

    # Print BERT embeddings
    #print(last_hidden_states.size())



>>>>>>> 0f0e8e7bffd20db1438c43edfd650580b466fdbf
