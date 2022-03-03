import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context='talk')



class Transformer(nn.Module):

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, mask):
        encoder_output = self.encoder(src, mask)
        out = self.decoder(trg, encoder_output)
        return out


class Encoder(nn.Module):
    def __init__(self, encoder_layer, n_layer): #n_layer는 Encoder layer의 개수. 즉, head의 개수
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_layer))

    def forward(self, x, mask):
        out = x
        for layer in self.layers:
            out = layer(out, mask)
        return out

class EncoderLayer(nn.Module):

    def __init__(self, multi_head_attention_layer, position_wise_feed_forward_layer):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention_layer = multi_head_attention_layer
        self.position_wise_feed_forward_layer = position_wise_feed_forward_layer

    def forward(self, x, mask):
        out = self.multi_head_attention_layer(query=x, key=x, value=x, mask=mask)
        out = self.positional_wise_feed_forward_layer(out)
        return out

def calculate_attention(self, query, key, value, mask):
    d_k = key.size(-1) #d_k 크기를 가져옴
    attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^t, |attention_score| = (n_batch, seq_len, seq_len)
    attention_score = attention_score / math.sqrt(d_k) # scailing

    if mask is not None:
        attention_score = attention_score.masked_fill(mask==0, -1e9) # masking

    attention_prob = F.softmax(attention_score, dim=-1) # |attention_prob| = (n_batch, seq_len, seq_len)
    out = torch.matmul(attention_prob, value) # |attention_prob x v| = (n_batch, seq_len, d_k)

    return out

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, h, qkv_fc_layer, fc_layer):
        # |qkv_fc_layer| = (d_embed, d_model)
        # |fc_layer| = (d_model, d_embed)

        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.d_model = d_model
        self.h = h
        self.query_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.key_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.value_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.fc_layer = fc_layer

        def forward(self, query, key, value, mask=None):
            # |q| = |k| = (n_batch, seq_len, d_embed)
            # |mask| = (n_batch, seq_len, seq_len)
            n_batch = query.shape[0]

            def transform(x, fc_layer): # reshape (n_batch, seq_len, d_embed) to (n_batch, h, seq_len, d_k)
                out = fc_layer(x) # |out| = (n_batch, seq_len, d_model)
                out = out.view(n_batch, -1, self.h, self.d_model//self.h) # |out| = (n_batch, seq_len, h, d_k) # numpy에서 reshape과 같은 역활
                out = out.transpose(1, 2) # |out| = (n_batch, h, seq_len, d_k)
                return out
            
            query = transform(query, self.query_fc_layer) # |query| = |k| = |value| = (n_batch, h, seq_len, d_k)
            key = transform(key, self.key_fc_layer)
            value = transform(value, self.value_fc_layer)

            if mask is not None:
                mask = mask.unsqueeze(1) # |mask| = (n_batch, 1, seq_len, seq_len) 

            out = self.calculate_attention(query, key, value, mask) # |out| = (n_batch, h, seq_len, d_k)
            out = out.transpose(1, 2) # |out| = (n_batch, seq_len, h, d_k) # 1번 index와 2번 index 자리 바꿈
            out = out.contiguous().view(n_batch, -1, self.d_model) # |out| = (n_batch, seq_len, d_model)
            out = self.fc_layer(out) # |out| = (n_batch, seq_len, d_embed)
            return out

            
class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, first_fc_layer, second_fc_layer):
        self.first_fc_layer = first_fc_layer
        self.second_fc_layer = second_fc_layer

    def forward(self, x):
        out = self.first_fc_layer(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.second_fc_layer(out)
        return out

        
