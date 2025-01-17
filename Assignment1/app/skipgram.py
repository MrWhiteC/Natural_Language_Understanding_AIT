import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import brown
import numpy as np
from collections import OrderedDict

news_corpus = brown.sents(categories=['news'])
word2index = {}

flatten = lambda l: [item for sublist in l for item in sublist]
vocabs = list(set(flatten(news_corpus))) 
vocabs.append('<UNK>')
word2index = {v:idx for idx, v in enumerate(vocabs)}
index2word = {v:k for k, v in word2index.items()}

class Skipgram(nn.Module):
    
    def __init__(self, voc_size, emb_size):
        super(Skipgram, self).__init__()
        self.embedding_center  = nn.Embedding(voc_size, emb_size)
        self.embedding_outside = nn.Embedding(voc_size, emb_size)
    
    def forward(self, center, outside, all_vocabs):
        center_embedding     = self.embedding_center(center)  #(batch_size, 1, emb_size)
        outside_embedding    = self.embedding_center(outside) #(batch_size, 1, emb_size)
        all_vocabs_embedding = self.embedding_center(all_vocabs) #(batch_size, voc_size, emb_size)
        
        top_term = torch.exp(outside_embedding.bmm(center_embedding.transpose(1, 2)).squeeze(2))
        #batch_size, 1, emb_size) @ (batch_size, emb_size, 1) = (batch_size, 1, 1) = (batch_size, 1) 

        lower_term = all_vocabs_embedding.bmm(center_embedding.transpose(1, 2)).squeeze(2)
        #batch_size, voc_size, emb_size) @ (batch_size, emb_size, 1) = (batch_size, voc_size, 1) = (batch_size, voc_size) 
        
        lower_term_sum = torch.sum(torch.exp(lower_term), 1)  #(batch_size, 1)
        
        loss = -torch.mean(torch.log(top_term / lower_term_sum))  #scalar
        
        return loss
    
def get_embed(model,word):
    try:
        index = [word]
        word = torch.LongTensor([word2index[word]])
    except:
        index = word2index['<UNK>']
        word = torch.LongTensor([word2index['<UNK>']])
        
    
    embed_c = model.embedding_center(word)
    embed_o = model.embedding_outside(word)
    embed   = (embed_c + embed_o) / 2
    
    return embed[0][0].item(), embed[0][1].item()


def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def find_similar_word(model,input):
    sim_list = {w:cosine_similarity(get_embed(model,input),get_embed(model,w)) for w in vocabs }
    sort_sim_list = sorted(sim_list.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)
    return sort_sim_list[:10]