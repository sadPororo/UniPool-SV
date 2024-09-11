""" The code reproduces the proposed methology from the paper, "Exploring wav2vec 2.0 on speaker verification and language identification."

[Paper Reference]
- Speaker verification model
    Fan, Zhiyun, Meng Li, Shiyu Zhou, and Bo Xu. 
      "Exploring wav2vec 2.0 on speaker verification and language identification." in Interspeech (2021). 
      doi: 10.21437/Interspeech.2021-1280    

- Additive Margin Softmax Loss (AMSoftmax)
    Wang, Feng, Jian Cheng, Weiyang Liu, and Haijun Liu.
      "Additive margin softmax for face verification." IEEE Signal Processing Letters 25, no. 7 (2018): 926-930.
      doi: 10.1109/LSP.2018.2822810

[Code Reference]
- Official code implementation: [ no public code implementation has been released ]
- AMSoftmaxHead: 
    https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch/blob/master/loss_functions.py
    https://www.kaggle.com/code/peter0749/additive-margin-softmax-loss-with-visualization

[Summay]
- Overall architecture
    * The model class 'ExploreWV2' comprises the Wav2Vec 2.0 backbone and Additive Margin Softmax (AMSoftmax) Loss head.
    
- Exploitation of pretrained backbone
    * The fisrt transformer block output is used from the Wav2Vec 2.0 (refers to "Sec. 3.3. Feasibility Analysis" of the paper)
    * while the forward path after the first transformer will be ignored.
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torchaudio
from torchaudio.models import wav2vec2_base, wav2vec2_large

from typing import Tuple, Optional


# Code Reference: https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch/blob/master/loss_functions.py
#                 https://www.kaggle.com/code/peter0749/additive-margin-softmax-loss-with-visualization
class AMSoftmaxHead(nn.Module):
    """ An implementation of "Additive Margin Softmax Loss"
    
    The original code is taken from cvqluu's git, but some parts are modified to resolve numerical issues.
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        super(AMSoftmaxHead, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        self.W.data.uniform_(-1, 1).renorm(2, 1, 1e-5).mul_(1e5)

    def forward(self, x, label):
        """
            x     (torch.Tensor.float) : [batch_size, embed_size]
            label (torch.Tensor.long)  : [batch_size] _categorical values
        """
        assert len(x) == len(label), "INPUT/TARGET batch size mismatch"
        assert torch.min(label).item() >= 0, f"There's a TARGET index < 0, index value:{torch.min(label).item()}"
        assert torch.max(label).item() < self.out_features, f"There's a TARGET index >= num. classes, index value:{torch.max(label).item()}"
        
        X = F.normalize(x, dim=1)
        W = self.W.renorm(2,1,1e-5).mul_(1e5)
        cossim = X.mm(W).clamp(-1, 1)
        
        numerator = self.s * (torch.diagonal(cossim.transpose(0, 1)[label]) - self.m)
        excl = torch.cat([torch.cat((cossim[i, :y], cossim[i, y+1:])).unsqueeze(0) for i, y in enumerate(label)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        
        return cossim, -torch.mean(L)


# Code Reference : https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L141
def _compute_mask_indices(shape: Tuple[int, int], mask_prob: float, mask_length: int, attention_mask: Optional[torch.LongTensor]=None, min_masks:int=0):
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape (Tuple[int, int]): The shape for which to compute masks. This should be of a tuple of size 2 where
                                    the first element is the batch size and the second element is the length of the axis to span.
        mask_prob (float): The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                                independently generated mask spans of length `mask_length` is computed by 'mask_prob*shape[1]/mask_length'. 
                                Note that due to overlaps, `mask_prob` is an upper bound and the actual percentage will be smaller.
        mask_length (int): size of the mask
        min_masks (int, optional): minimum number of masked spans. Defaults to 0.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask


class SVmodel(nn.Module):
    """ An implementation of the proposed speaker verification model from the paper,
            "Exploring wav2vec 2.0 on speaker verification and language identification."
    
    This is an unofficial reproduction of the proposal since there is no public code implementation has been released.
    """
    def __init__(self, config:dict):
        super(SVmodel, self).__init__()

        # Wav2Vec 2.0
        if config['model']['w2v_model'] == 'base':
            self.embed_size = 768
            if config['model']['use_pretrain']:
                self.w2v_encoder = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
            else:
                self.w2v_encoder = wav2vec2_base()
            
        elif config['model']['w2v_model'] == 'large':
            self.embed_size = 1024
            if config['model']['use_pretrain']:
                self.w2v_encoder = torchaudio.pipelines.WAV2VEC2_LARGE.get_model()
            else:
                self.w2v_encoder = wav2vec2_large()
            
        else:
            raise NotImplementedError(config['model']['w2v_model'])
        
        # SpecAugment
        if config.get('model').get('specaug') is not None:
            self.withaug = config['model']['specaug']
            self.mask_time_prob      = config['model']['mask_time_prob']
            self.mask_time_length    = config['model']['mask_time_length']
            self.mask_feature_prob   = config['model']['mask_feature_prob']
            self.mask_feature_length = config['model']['mask_feature_length']        
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(512).uniform_())
            
        else:
            self.withaug = False        
            self.mask_time_prob      = 0
            self.mask_time_length    = 0
            self.mask_feature_prob   = 0
            self.mask_feature_length = 0
            self.masked_spec_embed = None
        
        # Finetuning
        self.num_layers     = config['model']['num_layers']
        self.amsoftmax_loss = AMSoftmaxHead(self.embed_size, config['data']['nb_class_train'])
        
    # Code Reference : https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1478
    def _spec_augment(self, hidden_states:torch.Tensor):
        """ Mask extracted features along time axis and/or along feature axis
        
        Input:
            hidden_states (torch.Tensor.float) : [batch_size, max_lenth(time), embedding_size(feature)]
        """
        
        if self.withaug:
            B, T, E = hidden_states.size()
            
            if self.mask_time_prob > 0: # Time-axis Augmentation
                mask_indices = _compute_mask_indices(
                    shape = (B, T),
                    mask_prob=self.mask_time_prob,
                    mask_length=self.mask_time_length
                    )
                mask_indices = torch.tensor(mask_indices, device=hidden_states.device, dtype=torch.bool)
                hidden_states[mask_indices] = self.masked_spec_embed.to(hidden_states.dtype)
            
            if self.mask_feature_prob > 0: # Feature-axis Augmentation
                mask_indices = _compute_mask_indices(
                    shape=(B, E),
                    mask_prob=self.mask_feature_prob,
                    mask_length=self.mask_feature_length
                    )
                mask_indices = torch.tensor(mask_indices, device=hidden_states.device, dtype=torch.bool)
                mask_indices = mask_indices[:, None].expand(-1, T, -1)
                hidden_states[mask_indices] = 0
            
        else:
            pass
        
        return hidden_states
        
    def forward(self, waveforms, lengths, targets=None):
        """
        Input:
            waveforms (torch.Tensor.float) : [batch_size, max_length]
            lengths   (torch.Tensor.long)  : [batch_size] _lengths for each audio sample in the batch
            targets   (torch.Tensor.long)  : [batch_size] _categorical values
            
        Output:
            if train:
                logit (torch.Tensor.float) : [batch_size, nb_classes] _cosine similarity
                L     (torch.Tensor.float) : [1] _additive margin softmax loss
                
            else inference:
                outputs (torch.Tensor.float) : [batch_size] _cosine similarity
        """
        if self.training:
            # W2V2 Conv-layers
            outputs, lengths = self.w2v_encoder.feature_extractor(waveforms, lengths)
            
            # SpecAugment
            if self.withaug:
                outputs = self._spec_augment(outputs)
                
            # TF-layers
            outputs = self.w2v_encoder.encoder.extract_features(outputs, lengths, num_layers=self.num_layers)
            outputs = outputs[-1]
            B, T, E = outputs.size()
            
            # Mean-pooling
            mask = (torch.arange(T, device=outputs.device) < lengths[:, None]).unsqueeze(-1)
            outputs = (outputs * mask).sum(dim=1) / lengths.unsqueeze(-1) # (B, E)
            
            # AM-softmax loss head
            logit, L = self.amsoftmax_loss(outputs, targets)
            return logit, L
            
        else: # self.eval (batch-size = 1)
            # W2V2 Conv/TF-layers
            outputs, _ = self.w2v_encoder.feature_extractor(waveforms, length=None)
            outputs = self.w2v_encoder.encoder.extract_features(outputs, num_layers=self.num_layers)
            outputs = outputs[-1] # (B=1, T E)
            
            # Mean-pooling
            outputs = outputs.mean(dim=1) # (B=1, E)
            return F.normalize(outputs, dim=-1)
        
        
