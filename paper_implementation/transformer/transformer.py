import math
import torch
import torch.nn.functional as F
import torch.nn as nn


# Here is example how attention works.
## Theory can be access from here https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html#Scaled-Dot-Product-Attention

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Params:
        q: query is a feature vector that describes what we are looking for in the sequence.
        k: key which is again a feature vectore. (e keys should be designed such that we can identify the elements we want to pay attention to based on the query.)
        v: Feature vector is the one we want to average over.
    Return:
        (values, attention)
    """
    # Define dim_k match with dim query size.
    d_k = q.size()[-1]
    
    # do matrix mutlipication between q and k
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    
    # do some normalization here
    attn_logits = attn_logits / math.sqrt(d_k) # 1/ sqrt(dk)

    # The block Mask (opt.) in the diagram above represents the optional masking of specific entries in the attention matrix
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask==0, -9e15)
    
    # Get attention
    attention = F.softmax(attn_logits)

    # Matmul between result attention with feature vector
    values = torch.matmul(attention, v)
    return values, attention

# Multiple Head Attention

# Helper function to support different mask shapes.
# Output shape supports = (batch_size, number_of_heads, seq_length, seq length)
# If 2D: broadcasted over batch size and number of heads
# If 3D: broadcasted over number of heads
# If 4D: leave as is

def expand_mask_dim(mask: torch.Tensor):
    assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.__reset_params()
    
    def __reset_params(self):
        # Original transformer init weight
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        # Fill bias with 0
        self.qkv_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
    
    def forward(self, x, mask=None, return_attn=False):
        batch_size, seq_length, _ = x.size()

        if mask is not None:
            mask = expand_mask_dim(mask)

        # Forward q,k, and v with linear layer
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, head, seqlen, dim]
        q, k, v = qkv.chunk(3, dim=-1)

        # Calculate valute outputs
        values, attention = scaled_dot_product_attention(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)

        # linear output layer
        o = self.o_proj(values)
        if return_attn:
            return o, attention
        else:
            return o


class TransformerEncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()

        # Attention Layer
        self.self_attn = MultiHeadAttention(input_dim, input_dim, num_heads)

        # Two-Layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply norm in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Attention Part
        attn_out = self.self_attn(x, mask)
        # Residual part
        x += self.dropout(attn_out)
        x = self.norm1(x)

        # Feedforward part
        linear_out = self.linear_net(x)
        # residual connection
        x += self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderBlock(**block_args) for _ in range(num_layers)])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attn=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps


# Positional Encoding Part
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model : Hidden dimentionality of the inpyyt.
            max_len : max_len of sequence to expect.
        """
        super().__init__()

        # Create matrix of [seqlen, hiddendim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# TODO: Build Decoder Part

# TODO: Build Transformer Model

if __name__ == "__main__":
    seq_len, d_k = 3, 1
    q = torch.randn(seq_len, d_k)
    k = torch.randn(seq_len, d_k)
    v = torch.randn(seq_len, d_k)
    # print(f"Query size: {q.shape}")
    values, attention = scaled_dot_product_attention(q, k, v)
    print("Q\n", q)
    print("K\n", k)
    print("V\n", v)
    print("Values\n", values)
    print("Attention\n", attention)
    print(f"Attention shape {attention.shape}")

