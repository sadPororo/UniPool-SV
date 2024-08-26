import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2GumbelVectorQuantizer

import torchaudio
from transformers import Wav2Vec2ForPreTraining, Wav2Vec2Config
from torchaudio.models import wav2vec2_base, wav2vec2_large

from einops import rearrange, repeat, reduce
from typing import Optional, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2GumbelVectorQuantizer

import torchaudio
from einops import rearrange, repeat, reduce


def xavier_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: 
            torch.nn.init.zeros_(m.bias)
    
    if isinstance(m, torch.nn.Parameter) and len(m.size()) > 1:
        print(m)
        torch.nn.init.xavier_uniform_(m)


# https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L369
class Wav2Vec2PositionalConvEmbedding(nn.Module):
    def __init__(self, hidden_size=768, num_conv_pos_embeddings=128, num_conv_pos_embedding_groups=16):
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=num_conv_pos_embeddings,
            padding=num_conv_pos_embeddings // 2,
            groups=num_conv_pos_embedding_groups,
        )

        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm
        self.conv = weight_norm(self.conv, name="weight", dim=2)
            
        self.padding = Wav2Vec2SamePadLayer(num_conv_pos_embeddings)
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L408
class Wav2Vec2SamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class MHSA(nn.Module):
    """ Multi-headed Self Attention

    hidden_size (int): hidden dimension
    num_heads  (int): number of heads
    p_dropout  (float): attention dropout ratio, between [0, 1]
    """
    def __init__(self, hidden_size:int=768, num_heads:int=12, p_dropout:float=0.1):
        super(MHSA, self).__init__()
        assert hidden_size % num_heads == 0
        self.num_heads   = num_heads
        self.temperature = (hidden_size//num_heads) ** -.5 # 1 / sqrt(d_k)
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(p=p_dropout)
        self.o_proj  = nn.Linear(hidden_size, hidden_size, bias=True)
        self.apply(xavier_init)
        
    def forward(self, q, k, v, mask=None):
        """
            (q,k,v) : (B, T(q,k,v), hidden)
            mask    : (B, 1, T, T)
        """
        # (Q, K, V) projection
        Q = self.q_proj(q)
        K = self.k_proj(k)
        V = self.v_proj(v)
        
        # Multi-headed split: [B, T, E] --to-> [B, n_heads, T, d]
        Q = rearrange(Q, "B T (h d) -> B h T d", h=self.num_heads)
        K = rearrange(K, "B T (h d) -> B h d T", h=self.num_heads) # Transposed K
        V = rearrange(V, "B T (h d) -> B h T d", h=self.num_heads)
        
        # Scaled dot-product, W: [B, h, Tq, Tk]
        W = torch.matmul(Q, K)
        if mask is not None:
            W.masked_fill_(~mask, torch.finfo(torch.float32).min)
        W = W * self.temperature
        
        # Attention score, A: [B, h, Tq, Tk]
        A = torch.softmax(W, dim=-1)
        A = self.dropout(A)
        
        # Weighted sum of value projection, A x V: [B, h, Tq, Tk] x [B h Tk E] -> [B h Tq E]
        O = torch.matmul(A, V)
        O = rearrange(O, "B h T d -> B T (h d)")
        
        # Output projection
        O = self.o_proj(O)
        
        return O


class PFFN(nn.Module):
    """ Position-wise Feed Forward Network; 2-Layered Linear Transformation with GELU activation

    hidden_size (int): hidden dimension
    intermediate_size (int): intermediate hidden dimension
    p_dropout (float): dropout ratio for the intermediate layer, between [0, 1]
    """
    def __init__(self, hidden_size:int=768, intermediate_size:int=3072, p_dropout:float=0.0):
        super(PFFN, self).__init__()
        self.ffn_transformation = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=True),
            nn.GELU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(intermediate_size, hidden_size, bias=True),
        )
        self.apply(xavier_init)
    
    def forward(self, x):
        """ 
            x : (B, T, hidden)
        """
        return self.ffn_transformation(x)
        

class TFencoderLayer(nn.Module):
    """ A single Transformer encoder layer

    hidden_size (int): hidden dimension
    num_heads  (int): number of heads
    intermediate_size (int): intermediate hidden dimension    
    p_attn_drop (float): dropout ratio for the attention score matrix, between [0, 1]
    p_ffn_drop (float): dropout ratio for the intermediate layer from Feed Forward Network, between [0, 1]
    p_hidden_drop (float): dropout ratio for the output of Feed Forward Network, between [0, 1]
    """
    def __init__(self, hidden_size:int=768, num_heads:int=12, intermediate_size:int=3072, 
                    p_attn_drop:float=0.1, p_ffn_drop:float=0., p_hidden_drop:float=0.1):
        super(TFencoderLayer, self).__init__()
        
        self.mhsa = MHSA(hidden_size, num_heads, p_attn_drop)
        self.drop_attn = nn.Dropout(p=p_hidden_drop)
        self.norm_attn = nn.LayerNorm(hidden_size)
        
        self.pffn = PFFN(hidden_size, intermediate_size, p_ffn_drop)
        self.drop_pffn = nn.Dropout(p=p_hidden_drop)
        self.norm_pffn = nn.LayerNorm(hidden_size)
        
    def forward(self, x, mask=None):
        """
            x   : (B, T, hidden)
            mask: (B, 1, T, T)
            verbose: return attention score if True
        """            
        x = self.norm_attn(self.drop_attn(self.mhsa(x, x, x, mask)) + x)
        x = self.norm_pffn(self.drop_pffn(self.pffn(x)) + x)
        return x


class TFcrossEncoderLayer(nn.Module):
    """ A single Transformer encoder layer

    hidden_size (int): hidden dimension
    num_heads  (int): number of heads
    intermediate_size (int): intermediate hidden dimension    
    p_attn_drop (float): dropout ratio for the attention score matrix, between [0, 1]
    p_ffn_drop (float): dropout ratio for the intermediate layer from Feed Forward Network, between [0, 1]
    p_hidden_drop (float): dropout ratio for the output of Feed Forward Network, between [0, 1]
    """
    def __init__(self, hidden_size:int=768, num_heads:int=12, intermediate_size:int=3072, 
                    p_attn_drop:float=0.1, p_ffn_drop:float=0., p_hidden_drop:float=0.1):
        super(TFcrossEncoderLayer, self).__init__()
        
        self.mhsa = MHSA(hidden_size, num_heads, p_attn_drop)
        self.drop_attn = nn.Dropout(p=p_hidden_drop)
        self.norm_attn = nn.LayerNorm(hidden_size)
        
        self.pffn = PFFN(hidden_size, intermediate_size, p_ffn_drop)
        self.drop_pffn = nn.Dropout(p=p_hidden_drop)
        self.norm_pffn = nn.LayerNorm(hidden_size)
        
    def forward(self, x, c, mask=None):
        """
            x: (B, Tx, hidden) encoding target -> Query
            c: (B, Tc, hidden) given context -> Key, Value
            mask: (B, 1, Tx, Tc)
        """
        x = self.norm_attn(self.drop_attn(self.mhsa(x, c, c, mask)) + x)
        x = self.norm_pffn(self.drop_pffn(self.pffn(x)) + x)
        return x


class TFdecoderLayer(nn.Module):
    """ A single Transformer decoder layer
    
    embed_size (int): hidden dimension
    num_heads (int)
    """
    
    def __init__(self, hidden_size:int=768, num_heads:int=12, intermediate_size:int=3072, 
                    p_attn_drop:float=0.1, p_crss_drop:float=0.1, p_ffn_drop:float=0.0, p_hidden_drop:float=0.1):
        super(TFdecoderLayer, self).__init__()
        
        self.mhsa = MHSA(hidden_size, num_heads, p_attn_drop)
        self.drop_attn = nn.Dropout(p=p_hidden_drop)
        self.norm_attn = nn.LayerNorm(hidden_size)
        
        self.crss = MHSA(hidden_size, num_heads, p_crss_drop)
        self.drop_crss = nn.Dropout(p=p_hidden_drop)
        self.norm_crss = nn.LayerNorm(hidden_size)
        
        self.pffn = PFFN(hidden_size, intermediate_size, p_ffn_drop)
        self.drop_pffn = nn.Dropout(p=p_hidden_drop)
        self.norm_pffn = nn.LayerNorm(hidden_size)
        
    def forward(self, x, c, mask_self=None, mask_crss=None):
        """
            x: (B, Tx, E) _decoder embedding -> Query
            c: (B, Tc, E) _encoder context -> Key, Value
            mask_self: (B, 1, Tx, Tx)
            mask_crss: (B, 1, Tx, Tc)
        """
        x = self.norm_attn(self.drop_attn(self.mhsa(x, x, x, mask_self)) + x)
        x = self.norm_crss(self.drop_crss(self.crss(x, c, c, mask_crss)) + x)
        x = self.norm_pffn(self.drop_pffn(self.pffn(x)) + x)
        return x

    
class TFencoder(nn.Module):
    """ Transformer encoder """
    
    def __init__(self, num_layers:int=12, hidden_size:int=768, num_heads:int=12, intermediate_size:int=3072, 
                    p_attn_drop:float=0.1, p_ffn_drop:float=0.0, p_hidden_drop:float=0.1, p_layer_drop:float=0.0,
                    num_conv_pos_embeddings:int=128, num_conv_pos_embedding_groups:int=16):
        super(TFencoder, self).__init__()
        
        self.pos_embedding = Wav2Vec2PositionalConvEmbedding(hidden_size, num_conv_pos_embeddings, num_conv_pos_embedding_groups)
        self.tf_enc_layers = nn.ModuleList(
            [TFencoderLayer(hidden_size, num_heads, intermediate_size, p_attn_drop, p_ffn_drop, p_hidden_drop) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.p_layer_drop = p_layer_drop
        
    def forward(self, x, length=None):
        """
            x (B, T, hidden): encoding sequence
            length (B,): sequence length. Defaults to None.
        """
        # positional encoding
        x = x + self.pos_embedding(x)
        
        # encoder attention mask generation
        if length is not None:
            B, T, _ = x.size()
            mask = (torch.arange(T, device=x.device) < length[:, None]) # (B, T)
            mask = (mask.unsqueeze(-1).repeat(1, 1, T) & mask.unsqueeze(1).repeat(1, T, 1)).unsqueeze(1) # (B, 1, T, T)
        else:
            mask = None        
        
        # layer forward
        context = []
        for i in range(self.num_layers):
            if (i != self.num_layers-1) and torch.rand(1).item() < self.p_layer_drop:
                context.append(None)
            else:
                x = self.tf_enc_layers[i](x, mask)
                context.append(x)
                
        return context
    
    
class TFcrossEncoder(nn.Module):
    """ Transformer decoder """
    def __init__(self, num_layers:int=12, hidden_size:int=768, num_heads:int=12, intermediate_size:int=3072, 
                    p_attn_drop:float=0.1, p_ffn_drop:float=0.0, p_hidden_drop:float=0.1,
                    num_conv_pos_embeddings:int=128, num_conv_pos_embedding_groups:int=16):
        super(TFcrossEncoder, self).__init__()
        
        self.pos_embedding = Wav2Vec2PositionalConvEmbedding(hidden_size, num_conv_pos_embeddings, num_conv_pos_embedding_groups)
        self.crossenc_layers = nn.ModuleList(
            [TFcrossEncoderLayer(hidden_size, num_heads, intermediate_size, p_attn_drop, p_ffn_drop, p_hidden_drop) for _ in range(num_layers)]
        ) 

        self.num_layers = num_layers
        
    def forward(self, x, c, inp_length=None, ctx_length=None):
        """
            x (B, Tx, hidden): encoding target sequence
            c [n_layers x (B, Tc, hidden)]: context sequence
            inp/ctx_length (B,): input/context sequence length
        """
        # positional encoding
        x = x + self.pos_embedding(x)        
        
        # encoding sequence mask generation
        B, Tx, _ = x.size()
        if inp_length is not None:
            inp_mask = (torch.arange(Tx, device=x.device) < inp_length[:, None])
        else:
            inp_mask = torch.ones(size=(B, Tx), device=x.device).bool()
        
        # context sequence mask generation
        B, Tc, _ = c[-1].size()
        if ctx_length is not None:
            ctx_mask = (torch.arange(Tc, device=x.device) < ctx_length[:, None])
        else:
            ctx_mask = torch.ones(size=(B, Tc), device=x.device).bool()
        
        # cross attention mask
        attn_mask = (inp_mask.unsqueeze(-1).repeat(1, 1, Tc) & ctx_mask.unsqueeze(1).repeat(1, Tx, 1)).unsqueeze(1) # (B, 1, Tx, Tc)
                        
        # layer forward
        output = []
        for i in range(self.num_layers):
            if c[i] is not None:
                x = self.crossenc_layers[i](x, c[i], attn_mask)
                output.append(x)
                
        return output

    
class TFdecoder(nn.Module):
    """ Transformer decoder """
    def __init__(self, num_layers:int=12, hidden_size:int=768, num_heads:int=12, intermediate_size:int=3072, 
                    p_attn_drop:float=0.1, p_crss_drop:float=0.1, p_ffn_drop:float=0.0, p_hidden_drop:float=0.1,
                    num_conv_pos_embeddings:int=128, num_conv_pos_embedding_groups:int=16):
        super(TFdecoder, self).__init__()
        
        self.pos_embedding = Wav2Vec2PositionalConvEmbedding(hidden_size, num_conv_pos_embeddings, num_conv_pos_embedding_groups)
        self.tf_dec_layers = nn.ModuleList(
            [TFdecoderLayer(hidden_size, num_heads, intermediate_size, p_attn_drop, p_crss_drop, p_ffn_drop, p_hidden_drop) for _ in range(num_layers)]
        ) 

        self.num_layers = num_layers
        
    def forward(self, x, c, enc_length=None, dec_length=None):
        """
            x (B, Tx, E): decoder sequence
            c [n_layers x (B, Tc, E)]: encoded sequence
            enc/dec_length (B,): enc/decoder sequence length
        """
        # positional encoding
        x = x + self.pos_embedding(x)        
        
        # decoder self-attention mask generation
        B, Tx, _ = x.size()
        if dec_length is not None:
            dec_mask = (torch.arange(Tx, device=x.device) < dec_length[:, None])
        else:
            dec_mask = torch.ones(size=(B, Tx), device=x.device).bool()            
        mask_self = (dec_mask.unsqueeze(-1).repeat(1, 1, Tx) & dec_mask.unsqueeze(1).repeat(1, Tx, 1)).tril(diagonal=0).unsqueeze(1) # (B, 1, Tx, Tx) - True on lower triangular
        
        # cross-attention mask generation
        B, Tc, _ = c[-1].size()
        if enc_length is not None:
            enc_mask = (torch.arange(Tc, device=x.device) < enc_length[:, None])
        else:
            enc_mask = torch.ones(size=(B, Tc), device=x.device).bool()
        mask_crss = (enc_mask.unsqueeze(1).repeat(1, Tx, 1) & dec_mask.unsqueeze(-1).repeat(1, 1, Tc)).unsqueeze(1) # (B, 1, Tx, Tc)
        
        # layer forward
        output = []
        for i in range(self.num_layers):
            if c[i] is not None:
                x = self.tf_dec_layers[i](x, c[i], mask_self, mask_crss)
                output.append(x)
                
        return output
