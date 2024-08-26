#%%

import math
import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce

import torchaudio
from torchaudio.models import wav2vec2_base, wav2vec2_large, wav2vec2_large_lv60k

from transformers import Wav2Vec2Config, Wav2Vec2Model
from transformers import WavLMConfig, WavLMModel
from transformers import HubertConfig, HubertModel
# from vector_quantize_pytorch import VectorQuantize

# from submodules import *

import time
import matplotlib.pyplot as plt
# torch.autograd.set_detect_anomaly(True)

            
def xavier_init(*modules):
    for m in modules:
        torch.nn.init.xavier_uniform_(m)
        
def kaiming_init(*modules):
    for m in modules:
        torch.nn.init.kaiming_uniform_(m, nonlinearity='relu')
        
def zero_init(*modules):
    for m in modules:
        torch.nn.init.zeros_(m)

def apply_xavier_init(m:nn.Module):
    for name, p in m.named_parameters():
        if 'weight' in name: xavier_init(p)
        elif 'bias' in name: zero_init(p)

def apply_kaiming_init(m:nn.Module):
    for name, p in m.named_parameters():
        if 'weight' in name: kaiming_init(p)
        elif 'bias' in name: zero_init(p)
                

#%%


class MDConv2d(nn.Module):
    def __init__(self, conv_dim, kernel_size=(3,3), dilation=[1]):
        super().__init__()
        
        self.mdconv = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(conv_dim),
                nn.ReLU(),
                nn.Conv2d(conv_dim, conv_dim, kernel_size=kernel_size, dilation=d, padding='same')
            ) for _, d in enumerate(dilation)  
        ])
        for i, conv in enumerate(self.mdconv):
            apply_kaiming_init(conv[-1])
        self.out_dim = conv_dim
        
    def forward(self, x):
        """
        Input:
            x: [ d x (B C H W) ]
        Output:
            x:  (B C H W)
        """
        output = ()
        for i, conv in enumerate(self.mdconv):
            output += (conv(x[i]),) # [d x (B C H W)]
        output = torch.stack(output, dim=1).sum(dim=1) # (B d C H W).sum(dim=d) -> (B C H W)
        
        return output


class D2Block(nn.Module):
    def __init__(self, in_dim, conv_dim, kernel_size=(3,3), dilation=[1,2,4]):
        super().__init__()
        
        self.projection = nn.Conv2d(in_dim, conv_dim, kernel_size=(1,1))
        apply_kaiming_init(self.projection)
        
        self.dense = nn.ModuleList([
            MDConv2d(conv_dim, kernel_size, dilation[:(i+1)]) for i, _ in enumerate(dilation)
        ])
        self.out_dim = conv_dim * (len(dilation) + 1)
        
    def forward(self, x):
        """
        Input:
            x: (B C H W)
        Output:
            x: (B [conv_dim x d] H W)
        """
        x = self.projection(x) # (B C H W) -> (B conv_dim H W)
        
        output = (x,)
        for i, layer in enumerate(self.dense):
            output += (layer(output),)
        output = torch.cat(output, dim=1) # (B [conv_dim x d] H W)
        
        return output


class ChannelAttention(nn.Module):
    def __init__(self, in_dim, bottleneck):
        super().__init__()
        
        self.avg_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_dim, bottleneck//2, kernel_size=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(bottleneck//2),
            nn.Conv2d(bottleneck//2, in_dim, kernel_size=(1,1))
        ); 
        apply_kaiming_init(self.avg_se[1])
        apply_xavier_init(self.avg_se[-1])

        self.max_se = nn.Sequential(
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Conv2d(in_dim, bottleneck//2, kernel_size=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(bottleneck//2),
            nn.Conv2d(bottleneck//2, in_dim, kernel_size=(1,1))
        )
        apply_kaiming_init(self.max_se[1])
        apply_xavier_init(self.max_se[-1])
        
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        """
        Input/Output:
            x: (B, in_dim,  L, T)
        """
        avg_pool = self.avg_se(x) # (B in_dim 1 1)
        max_pool = self.max_se(x) # (B in_dim 1 1)
        
        scale = self.act(avg_pool + max_pool) # (B in_dim 1 1)
        
        return x * scale


class SUPERB(nn.Module):
    def __init__(self, n_layers:int, i=0):
        super().__init__()
        self.score_vector = torch.zeros(n_layers)
        self.score_vector[i] = 1.
        self.score_vector = nn.Parameter(self.score_vector)
        # self.score_vector = nn.Parameter(torch.ones(n_layers) / n_layers)
        
    def forward(self, x:torch.Tensor):
        """
        Input:
            x: (B, D, L, T)
        Output:
            x: (B, D, T)
        """
        layer_weight = self.score_vector.softmax(dim=-1)
        x = x * layer_weight[None, None, :, None]
        x = x.sum(dim=2)
        
        return x


class VADscale(nn.Module):
    def __init__(self, in_dim:int, n_layers:int):
        super().__init__()

        self.superb = nn.ModuleList([SUPERB(n_layers, i) for i in range(n_layers)])
        self.weight = nn.Parameter(torch.randn(n_layers, in_dim, 1))
        self.bias   = nn.Parameter(torch.zeros(n_layers, 1, 1))
        xavier_init(self.weight); zero_init(self.bias)
        
    def forward(self, x, mask=None):
        """
        Input:
            x: (B, D, L, T)
            mask: (B, T)
        Output:
            x: (B, D, L, T)
        """
        x = torch.stack([self.superb[i](x) for i in range(x.size(2))], dim=2) # (B D L T)
        
        vad_scr = torch.matmul(x.permute(0,2,3,1), self.weight) + self.bias # B D L T -> B L T D -> B L T 1
        if mask is not None:
            vad_scr.masked_fill_(~mask[:, None, :, None], torch.finfo(torch.float32).min)
        vad_scr = vad_scr.sigmoid().permute(0,3,1,2) # B 1 L T
        
        return x * vad_scr

class ResConnection(nn.Module):
    def __init__(self, m):
        super().__init__()
        
        self.m = m
    def forward(self, x):
        return x + self.m(x)

class D3Block(nn.Module):
    def __init__(self, in_dim, conv_dim, kernel_size=(3,3), n_blocks=3, n_layers=1):
        super().__init__()
        block_dim = conv_dim * 4
        
        self.projection = nn.Sequential(
            nn.Conv2d(in_dim, block_dim, kernel_size=(1,1)),
            nn.BatchNorm2d(block_dim)
        ); apply_xavier_init(self.projection[0])
        
        self.vadnet = nn.ModuleList([
            VADscale(block_dim, n_layers) for _ in range(n_blocks)
        ])
        
        self.densenet = nn.ModuleList([
            ResConnection(
                nn.Sequential(
                    D2Block(block_dim, conv_dim, kernel_size),
                    ChannelAttention(block_dim, block_dim//2)
                )
            ) for _ in range(n_blocks)
        ])
        self.out_dim = block_dim * n_blocks
        
    def forward(self, x, mask=None):
        """
        Input:
            x: (B, in_dim,  L, T)
        Output:
            x: (B, in_dim,  L, T)
        """
        x = self.projection(x)
        
        output = ()
        for i, (vad, dense) in enumerate(zip(self.vadnet, self.densenet)):
            x = vad(x, mask)
            x = x + dense(x)
            output += (x,)
        
        output = torch.cat(output, dim=1) # B [n_blocks x block_dim] L T

        return output


class LayerAttentionPool(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers=1, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        d_k = out_dim // n_heads
        
        self.in_projection = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, (self.n_heads * d_k), kernel_size=(1, 1), bias=False)
        ); apply_xavier_init(self.in_projection[-1])
        
        self.W1 = nn.Parameter(torch.randn(self.n_heads, n_layers, n_layers//2))
        self.b1 = nn.Parameter(torch.zeros(n_heads, 1, n_layers//2))
        kaiming_init(self.W1); zero_init(self.b1)

        self.W2 = nn.Parameter(torch.randn(self.n_heads, n_layers//2, n_layers))
        self.b2 = nn.Parameter(torch.zeros(n_heads, 1, n_layers))
        xavier_init(self.W2), zero_init(self.b2)

        self.out_projection = nn.Sequential(
            nn.Conv1d((self.n_heads * d_k), out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_dim)
        ); apply_xavier_init(self.out_projection[0])
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        """ 
        """
        B = x.size(0)
        
        x = self.in_projection(x)
        x = rearrange(x, 'B (h k) L T -> (B T) (h L) k', h=self.n_heads) # ([B x T], [h x C], k)
        
        avg_pool = rearrange(F.adaptive_avg_pool1d(x, 1), "BT (h L) 1 -> BT h 1 L", h=self.n_heads)
        max_pool = rearrange(F.adaptive_max_pool1d(x, 1), "BT (h L) 1 -> BT h 1 L", h=self.n_heads) # ([B x T], h, 1, L)

        avg_pool_bnk = torch.matmul(F.relu((torch.matmul(avg_pool, self.W1) + self.b1)), self.W2) + self.b2
        max_pool_bnk = torch.matmul(F.relu((torch.matmul(max_pool, self.W1) + self.b1)), self.W2) + self.b2 # ([B x T], h, 1, C)
        
        pool_sum = avg_pool_bnk + max_pool_bnk
        attn_scr = self.dropout( rearrange(pool_sum.sigmoid(), "BT h 1 L -> BT (h L) 1") ) # ([B x T], [h x L], 1)
        x = rearrange(x * attn_scr, 'BT (h L) k -> BT (h k) L', h=self.n_heads) # ([B x T], [h x L], k) -> ([B x T], [h x k], L)
        
        output = F.adaptive_max_pool1d(x, 1) # ([B * T], [h * k], 1)
        output = rearrange(output, '(B T) hk 1 -> B hk T', B=B)
        output = self.out_projection(output) # (B D T)
        
        # attn_scr = rearrange(attn_scr, '(B T) (h L) 1 -> B h L T', B=B, h=self.n_heads)
        
        return output


class AttentiveStatisticPool(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        d_k = (in_dim * 3) // n_heads        
        
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim * 3, in_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(in_dim // 2),
            # nn.Tanh(), # I add this layer
            nn.Conv1d(in_dim // 2, in_dim, kernel_size=1),
            )
        self.dropout = nn.Dropout(p=dropout)
        self.attn_norm = nn.BatchNorm1d(in_dim * 2)
        
        self.out_proj = nn.Linear(in_dim * 2, out_dim)
        self.out_norm = nn.BatchNorm1d(out_dim)
        
    def forward(self, x:torch.Tensor, mask=None):
        """ 
        Input:
            x: (B C1 T) - float
            mask: (B T) - bool
        Output:
            x: (B C2)
        """
        T = x.size(-1)
        
        if mask is not None:
            mask = mask[:, None, :] # (B 1 T)
            N    = mask.sum(dim=-1, keepdim=True) # (B 1 1)
            mu = (x * mask).sum(dim=-1, keepdim=True) / N # (B C1 1)
            sg = torch.sqrt((((x - mu) ** 2) * mask).sum(dim=-1, keepdim=True) / N) # (B C1 1)
        else:
            mu = x.mean(dim=-1, keepdim=True) # (B C1 1)
            sg = x.std(dim=-1, keepdim=True)  # (B C1 1)
            
        stat_pool = torch.cat([x, mu.expand(-1,-1,T), sg.expand(-1,-1,T)], dim=1) # (B 3C T)
        
        attn_scr = self.attention(stat_pool) # (B C1 T)
        if mask is not None:
            attn_scr.masked_fill_(~mask, torch.finfo(torch.float32).min) # mask: (B 1 T)
        attn_scr = self.dropout(F.softmax(attn_scr, dim=-1)) # (B C1 T)
        
        attn_mu = torch.sum(x * attn_scr, dim=-1) # (B C1)
        attn_sg = torch.sqrt((torch.sum((x**2) * attn_scr, dim=-1) - attn_mu**2).clamp(min=1e-4)) # (B C1)
        attn_pool = torch.cat([attn_mu, attn_sg], dim=1) # (B 2xC1)
        attn_pool = self.attn_norm(attn_pool)
        
        x = self.out_norm(self.out_proj(attn_pool))
        return x


class Multiheaded_AAMsoftmax(nn.Module):
    def __init__(self, in_dim:int, n_heads:int, n_class:int, m:float, s:float, d_k=None):
        super().__init__()
        
        # local-variables
        self.m = m
        self.s = s        
        self.n_heads = n_heads
        self.n_class = n_class
        if d_k is None: 
            assert in_dim % self.n_heads == 0
            d_k = in_dim // n_heads
        
        # setup for angular additive marginal softmax
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m
        
        self.weight = nn.Parameter(torch.rand(self.n_heads, d_k, self.n_class))
        xavier_init(self.weight)
        
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, x, label):
        """
        Input:
            x: (B, D)
            label: (B,)
        Output:
            output: (B, C)
            loss: (1,)
        """
        # head-wise split & inter-vector cosine/sine -> angular measure
        x = rearrange(x, 'B (h k) -> B h 1 k', h=self.n_heads)
        cosine = torch.matmul(F.normalize(x, dim=-1),
                              F.normalize(self.weight, dim=1)).squeeze(2) # (B, h, C)
        sine = torch.sqrt( (1.0 - torch.mul(cosine, cosine)).clamp(0, 1) )
        phi  = cosine * self.cos_m - sine * self.sin_m
        phi  = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        
        # one-hot label for each heads
        label = label[:, None].expand(label.size(0), self.n_heads) # (B, h)
        one_hot = torch.zeros_like(cosine) # (B, h, C)
        one_hot.scatter_(-1, label[..., None], 1) # scatter value-1 on c-th class
        
        # calculate loss with batchfying the heads
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) # (B, h, C)
        output = output * self.s
        loss = self.ce(output.reshape(-1, self.n_class), label.reshape(-1)) # ([B x h], C)
        
        # get predicted class probability
        with torch.no_grad():
            output = F.softmax(output, dim=-1).mean(dim=1) # (B, h, C).mean(dim=h) -> (B, C)
            
        return output, loss


#%%

class SVmodel(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        
        # [HYPERPARAMS] _______________________________________________________________________________________________________________________
        # backbone
        self.BACKBONE_CFG = config['model']['backbone_cfg']
        self.USE_PRETRAIN = config['model']['use_pretrain']
        self.FRZ_PRETRAIN = config['model']['frz_pretrain']
        
        # D3Block
        self.N_LAYERS = config['model']['n_layers']
        self.N_BLOCKS = config['model']['n_blocks']
        self.CONV_DIM = config['model']['conv_dim']
        
        # AttnPool
        self.POOL_DIM = config['model']['pool_dim']

        # Loss
        self.NUM_TRAIN_CLASS = config['data']['nb_class_train']
        self.NUM_LOSS_HEADS  = config['model']['num_loss_heads']

        # [SUB-MODULES] ________________________________________________________________________________________________________________________
        # backbone
        if 'wavlm' in self.BACKBONE_CFG:
            self.backbone = WavLMModel.from_pretrained(self.BACKBONE_CFG) \
                if self.USE_PRETRAIN else WavLMModel(WavLMConfig.from_pretrained(self.BACKBONE_CFG))

        elif 'wav2vec2' in self.BACKBONE_CFG:
            self.backbone = Wav2Vec2Model.from_pretrained(self.BACKBONE_CFG) \
                if self.USE_PRETRAIN else Wav2Vec2Model(Wav2Vec2Config.from_pretrained(self.BACKBONE_CFG))
        
        elif 'hubert' in self.BACKBONE_CFG:
            self.backbone = HubertModel.from_pretrained(self.BACKBONE_CFG) \
                if self.USE_PRETRAIN else HubertModel(HubertConfig.from_pretrained(self.BACKBONE_CFG))
        else:
            raise NotImplementedError(self.BACKBONE_CFG)
        
        self.N_LAYERS = self.backbone.config.num_hidden_layers if self.N_LAYERS==0 else self.N_LAYERS
        if self.FRZ_PRETRAIN: self.freeze_backbone_modules_()
        
        # speaker representation encoder
        d3net = D3Block(
                in_dim=self.backbone.config.hidden_size,
                conv_dim=self.CONV_DIM,
                n_layers=self.N_LAYERS,
                n_blocks=self.N_BLOCKS
            )
        l_pool = LayerAttentionPool(
                in_dim=d3net.out_dim,
                out_dim=self.CONV_DIM * 4,
                n_heads=4,
                n_layers=self.N_LAYERS
            )
        t_pool = AttentiveStatisticPool(
                self.CONV_DIM * 4,
                self.POOL_DIM,
                n_heads=4
            )
        self.encoder = nn.ModuleDict({
            'd3net': d3net,
            'l_pool': l_pool,
            't_pool': t_pool
        })
        
        # Loss head
        self.aam_softmax = Multiheaded_AAMsoftmax(
            in_dim=self.POOL_DIM,
            n_heads=self.NUM_LOSS_HEADS,
            n_class=self.NUM_TRAIN_CLASS,
            m=0.2, s=30
        )

        
        # [AUGMENTATION] _____________________________________________________________________________________________________________________
        # Span-out
        self.SPANOUT = config['model']['spanout']
        self.SPANOUT_PROB = config['model']['spanout_prob']
        self.SPAN_MASK_PROB = config['model']['span_mask_prob']
        self.SPAN_MASK_LENGTH = config['model']['span_mask_length']
        
        span_mask_kernel = torch.ones(1, 1, self.SPAN_MASK_LENGTH)
        self.register_buffer('span_mask_kernel', span_mask_kernel)
        
    def freeze_backbone_modules_(self):
        self.backbone.eval()
        for p in self.backbone.parameters(): p.requires_grad_(False)            

    def forward(self, x:torch.Tensor, length:torch.LongTensor=None, target:torch.LongTensor=None):
        """ 
        Input: 
            x: (B, T)
            length: (B,)
            target: (B,)
        """
        # [Backbone] ______________________________________________________________________________________________        
        if length is not None:
            length = self.backbone._get_feat_extract_output_lengths(length)
            attn_mask = torch.arange(length.max().item(), device=x.device) < length[:, None]
        else:
            length = (torch.ones(x.size(0)) * self.backbone._get_feat_extract_output_lengths(x.size(1))).to(int)
            attn_mask = None
        
        output = self.backbone.forward(x, attention_mask=attn_mask, output_hidden_states=True)
        x = rearrange(torch.stack(output.hidden_states), 'L B T D -> B D L T')[:, :, 1:self.N_LAYERS+1, :]
        
        # [SpanOut] ______________________________________________________________________________________________
        if self.training:
            x, length = self.spanout_(x, length) # (B, D, L, T >> T')
            
        # [Speaker Encoder] ______________________________________________________________________________________

        # ATTN VAD
        attn_mask = torch.arange(x.size(-1), device=x.device) < length[:, None] if self.training else None
        # x, vad_scale = self.encoder['attn_vad'](x, attn_mask) # (B, D, L, T)
        
        # D2 Block
        x = self.encoder['d3net'](x, attn_mask) # (B, D->D', L, T)

        # Layer-attention pooling
        x = self.encoder['l_pool'](x) # (B, D, T)
                
        # Layer-attention pooling
        x = self.encoder['t_pool'](x, attn_mask) # (B, D, T)
        
        # [Loss Head] ____________________________________________________________________________________________
        if self.training:
            cls_pred, L = self.aam_softmax(x, target)
            
            # return cls_pred, L + commit_loss
            return cls_pred, L
        
        else:
            x = rearrange(x, 'B (h k) -> B h k', h=self.NUM_LOSS_HEADS)
            
            return F.normalize(x.squeeze(0), p=2, dim=-1)

    def forward_check_(self, B:int):
        # Training forward
        print('[training forward check]')
        self.train()
        target = torch.randint(low=0, high=self.NUM_TRAIN_CLASS, size=(B,))
        length = (torch.rand(B) * 10 * 16000).to(int)
        print('- length:', length / 16000)
        print('- target:', target)
        
        x = torch.rand(B, length.max().item())
        x = F.layer_norm(x, x.size())
        print('- input:', x.size())
        
        cls_pred, L = self.forward(x, length, target)
        print('class prediction:', cls_pred.size())
        print('loss:', L, '\n')
        
        # Evaluation forward
        print('[evaluation forward check]')
        self.eval()
        x = torch.rand(1, length.max().item())
        x = F.layer_norm(x, x.size())
                
        x = self.forward(x)
        print('embedding output:', x.size())
        

    def spanout_(self, x:torch.Tensor, length:torch.LongTensor):
        """ 
        Input:
            x: (B, D, L, T)
            length: (B,)
        Output:
            x: (B, D, L, T') , where T' <= T
            length: (B,)
        """
        if self.SPANOUT:
            B, D, L, T = x.size()
            with torch.no_grad():
                # valid sequence mask: (B, T)
                pad_mask = torch.arange(T, device=x.device) < length[:, None]
                
                # partially span-dropped sequence mask: (B, T)
                spn_mask = torch.rand_like(pad_mask.float(), device=pad_mask.device) < self.SPAN_MASK_PROB # (B, T): randomly selected indice mask (the start of the span to drop)
                spn_mask = F.conv1d(spn_mask.flip(1).unsqueeze(1).float(), 
                                    self.span_mask_kernel, 
                                    padding=self.SPAN_MASK_LENGTH-1).squeeze(1).bool().flip(1)[:,:T] # (B, T): span-mask to drop the representations
                spn_mask = pad_mask & ~spn_mask
                
                # instance-wise augmentation appliance mask: (B,)
                keep_seq = torch.rand(B) > self.SPANOUT_PROB
                spn_mask[keep_seq] = pad_mask[keep_seq]
                
                # the final mask, after concatenating the valid spans: (B, T')
                new_mask = torch.arange(T, device=x.device) < spn_mask.sum(dim=1, keepdim=True)
            
                # __________________________________________________________________________________________________________________________________    
                x = x.masked_scatter_(new_mask[:, None, None, :].expand(-1, D, L, -1), x[spn_mask[:, None, None, :].expand(-1, D, L, -1)])
                length = new_mask.sum(dim=-1)
                
                x = x[..., :length.max().item()]
            
        return x, length




# ########################################################################################################################################################
    

# if __name__ == "__main__":
    # config = {
    #     'model':{'w2v_model_sz': 'base',
    #             'use_pretrain': True,
    #             'frozen_w2v': True,
    #             'num_tflayers': 4},
    #     'data':{'nb_class_train': 72}
    # }

# config = {}
# config['model'] = hypload('/home/jinsob/SV-TTS/19_src/benchmarks/ZFK-MDC9-BN/model-config.yaml')
# config['model']['backbone_cfg'] = 'facebook/wav2vec2-base'
# config['model']['n_layers'] = 12
# config['model']['conv_dim'] = 128
# config['model']['pool_dim'] = 192
# config['model']['span_out'] = False


# for dataset_nm in ['VCTK', 'Vox1-Base', 'LibriSpeech', 'Vox2-Base']:
#     config['data'] = hypload(f'/home/jinsob/SV-TTS/configs/data/data-{dataset_nm}-config.yaml')
#     model = SVmodel(config)
        
#     print(f'\n=== {dataset_nm} ===')
#     print('in Total  : {:.02f} M'.format(checkNumParams(model) / 1000000))
#     print('bakcbone  : {:.02f} M'.format(checkNumParams(model.backbone) / 1000000))
#     print('pooling   : {:.02f} M'.format(checkNumParams(model.encoder) / 1000000))
#     print('loss head : {:.02f} K'.format(checkNumParams(model.aam_softmax) / 1000))
#     break
# print()
# model.freeze_backbone_modules_()
# model.forward_check_(4)


# %%

