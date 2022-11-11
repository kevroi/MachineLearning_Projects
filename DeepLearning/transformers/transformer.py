import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """
    
    |  Add & Normalise  |
             ^
             |<--------------
    |                 |     ^
    |   Multi-Head    |     ^
    |   Attention     |     ^
    |                 |     ^
        ^   ^   ^           ^
                  -----------
    
    
    """
    def __init__(self, embedding_size, heads):
        """
        Args:
            embedding_size (int): Dimensionality of Embedding
            heads (int): Number of parts we want to split embedding into
        """
        super(SelfAttention, self).__init()
        # to handle incoming embedding
        self.embedding_size = embedding_size
        self.heads = heads
        self.head_dim = embedding_size//heads
        assert (self.head_dim * heads == embedding_size), "Embedding size must be divisble by number of heads"

        # linear layers for values, keys and queries
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fulcon_out = nn.Linear(self.heads*self.head_dim, self.embedding_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0] # no. examples sent in in parallel
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split embedding
        values = values.reshape(N, value_len, self.heads, self.head_dim) # this was prev of shape N * value_len * embedding_size
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        # multiply Queries with Keys, across the head_dim size dimension/axis
        # NOTE: the dimension we're multiplying across disapppears from the product
        # query shape: N * query_len * heads * head_dim
        # key shape: N * key_len * heads * head_dim
        # energy shape: N * heads * query_len * key_len
        # matrix mul using einstein summation - its so much easier for higher dim mults!
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys]) 

        if mask is not None:
            # mask will be an upper triangular matrix
            energy = energy.masked_fill(mask==0, float("-inf"))

        attention = torch.softmax(energy/(self.embedding_size**0.5), dim=3) # normalise probs across the dim of size key_len
        
        # multiplyattention by values, again using einsum, across the key_len size dimension/axis
        # attention shape: N * heads * query_len * key_len
        # key shape: N * value_len * heads * head_dim # key_len and value_len is always gonna be the same
        # energy shape: N * query_len * heads * head_diim
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        # reshape is to immediately do the next conctattenstion step (i.e. flatten last two dims)

        out = self.fulcon_out(out) # fully connected, no change of shape

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embedding_size=embedding_size, heads=heads)
        self.norm1 = nn.LayerNorm(embedding_size) # takes avg for every example
        self.norm2 = nn.LayerNorm(embedding_size)

        # feed forward neural network
        self.feed_forward = nn.Sequential(
                                            nn.Linear(embedding_size, forward_expansion*embedding_size),
                                            nn.ReLU(),
                                            nn.Linear(forward_expansion*embedding_size, embedding_size)
                                        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention+query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x)) # skip connection
        return out