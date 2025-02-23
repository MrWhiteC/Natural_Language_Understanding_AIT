from flask import Flask, request, jsonify, render_template
import datasets
from datasets import load_dataset
import math
import re
from   random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from bert import BERT
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

dataset = load_dataset('roneneldan/TinyStories', split='train[:1%]')

sentences = dataset['text']
text = [x.lower() for x in sentences] #lower case
text = [re.sub("[.,!?\\-]", '', x) for x in text] #clean all symbols

# Combine everything into one to make vocab
word_list = list(set(" ".join(text).split()))
word2id = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}  # special tokens

# Create the word2id in a single pass
for i, w in enumerate(word_list):
    word2id[w] = i + 4  # because 0-3 are already occupied

# Precompute the id2word mapping (this can be done once after word2id is fully populated)
id2word = {v: k for k, v in word2id.items()}
vocab_size = len(word2id)



n_layers = 12    # number of Encoder of Encoder Layer
n_heads  = 12    # number of heads in Multi-Head Attention
d_model  = 768  # Embedding Size
d_ff = d_model * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2
max_len = 700
max_mask   = 5


model = BERT(
    n_layers, 
    n_heads, 
    d_model, 
    d_ff, 
    d_k, 
    n_segments, 
    vocab_size, 
    max_len, 
    device
).to(device)

model.load_state_dict(torch.load('sentence_classification.pth'))
model.eval()


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/similarity', methods=['POST'])
def similarity():

    # try:
    data = request.json
    sentence_a = data.get('worda')
    sentence_b = data.get('wordb')
    # sentence_a = 'Your contribution helped make it possible for us to provide our students with a quality study.'
    # sentence_b = "Your contribution were of no help with our student's learn."

    text_a = sentence_a.lower()#lower case
    text_a = re.sub("[.,!?\\-]", '', text_a) #clean all symbols

    text_b = sentence_b.lower()#lower case
    text_b = re.sub("[.,!?\\-]", '', text_b) #clean all symbols



    token_list_a = []

    # Process sentences more efficiently

    token_list_a.append([word2id[word] for word in text_a.split()])

    token_list_b = []

    # Process sentences more efficiently

    token_list_b.append([word2id[word] for word in text_b.split()])



    input_ids_a = [word2id['[CLS]']] + token_list_a[0] + [word2id['[SEP]']]
    input_ids_b = [word2id['[CLS]']] + token_list_b[0] + [word2id['[SEP]']]

    segment_ids_a = [0] * (1 + len(token_list_a[0]) + 1)
    segment_ids_b = [0] * (1 + len(token_list_b[0]) + 1)

    cand_maked_pos = [i for i, token in enumerate(input_ids_a) if token != word2id['[CLS]'] and token != word2id['[SEP]']]
    shuffle(cand_maked_pos)
    masked_pos_a = []
    for pos in cand_maked_pos[:5]:
            masked_pos_a.append(pos)  

    n_pad = max_len - len(input_ids_a)
    input_ids_a.extend([0] * n_pad)
    segment_ids_a.extend([0] * n_pad)

    if max_mask > 5:
            n_pad = max_mask - 5
            masked_pos_a.extend([0] * n_pad)


    cand_maked_pos = [i for i, token in enumerate(segment_ids_b) if token != word2id['[CLS]'] and token != word2id['[SEP]']]
    shuffle(cand_maked_pos)
    masked_pos_b = []
    for pos in cand_maked_pos[:5]:
            masked_pos_b.append(pos)  

    n_pad = max_len - len(input_ids_b)
    input_ids_b.extend([0] * n_pad)
    segment_ids_b.extend([0] * n_pad)

    if max_mask > 5:
            n_pad = max_mask - 5
            masked_pos_b.extend([0] * n_pad)
    
    input_ids_a = torch.LongTensor(input_ids_a).to(device)
    segment_ids_a = torch.LongTensor(segment_ids_a).to(device)
    masked_pos_a = torch.LongTensor(masked_pos_a).to(device)

    input_ids_b = torch.LongTensor(input_ids_b).to(device)
    segment_ids_b = torch.LongTensor(segment_ids_b).to(device)
    masked_pos_b = torch.LongTensor(masked_pos_b).to(device)

    result_a,_ = model(input_ids_a.unsqueeze(0), segment_ids_a.unsqueeze(0), masked_pos_a.unsqueeze(0))  
    result_b,_ = model(input_ids_b.unsqueeze(0), segment_ids_b.unsqueeze(0), masked_pos_b.unsqueeze(0))  

    similarity_score = cosine_similarity(result_a.reshape(1, -1).detach().cpu(), result_b.reshape(1, -1).detach().cpu())[0, 0]

    if np.round(similarity_score) > 0 :
        similar_word = 'Entailment'
    elif np.round(similarity_score) == 0:
        similar_word = 'Neutral'
    else:
        similar_word = 'Contradiction'


    if not sentence_a:
        return jsonify({"error": "Word is required"}), 400

    # Replace the following line with your model's similarity computation logic
    # for token in output_max:
    #  print(mapping[token.item()],file=sys.stderr)


    return jsonify({
        "input_worda": sentence_a,
        "input_wordb": sentence_b,
        "similar_word": similar_word,
        "score": str(similarity_score),
    })
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)



