import torch
from torch import nn
import torch.nn.functional as F
import math

class ModelArch(nn.Module):
    def __init__(self, bert, heads):
        super(ModelArch, self).__init__()
        self.bert = bert 
        self.d_model = 768
        self.attention_1 = MultiheadAttention(heads, d_model=self.d_model, dropout=0.1)
        self.attention_2 = MultiheadAttention(heads, d_model=self.d_model, dropout=0.1)
        self.attention_3 = MultiheadAttention(heads, d_model=self.d_model, dropout=0.1)
        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)
        self.dropout_3 = nn.Dropout(0.1)
        self.norm_1 = Norm(self.d_model)
        self.norm_2 = Norm(self.d_model)
        self.norm_3 = Norm(self.d_model)
        self.norm_4 = Norm(self.d_model)
        self.softmax = nn.LogSoftmax(dim=1)
        self.gru = nn.GRU(self.d_model, 512, 1)
        self.fc = nn.Linear(512,4)

    def forward(self, sent_id, mask):
        outs = self.bert(sent_id, attention_mask=mask)
        embd = outs[0]
        x = self.norm_1(embd)
        x = x + self.dropout_1(self.attention_1(x, x, x))
        x = self.norm_2(x)
        x = x + self.dropout_2(self.attention_2(x, x, x))
        x = self.norm_3(x)
        x = x + self.dropout_3(self.attention_3(x, x, x))
        x = self.norm_4(x)
        x, _ = self.gru(x)
        x = F.avg_pool1d(x.transpose(2,1), kernel_size=50).squeeze()
        x = self.fc(x)
        x = self.softmax(x)
        return x


class MultiheadAttention(nn.Module):  
    def __init__(self, heads, d_model, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        '''concatenate heads and put through linear layer'''
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.size = d_model

        '''create two learnable parameters to calibrate normalisation'''
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(-2)
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)          
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output
