""" This code reproduces the proposal from the paper, "Fine-tuning wav2vec2 for speaker recognition."

[Paper Reference]
- Speaker verification model
    Nik Vaessen, and David A. Van Leeuwen.
      "Fine-tuning wav2vec2 for speaker recognition." in IEEE ICASSP (2022).
      doi: 10.1109/ICASSP43922.2022.9746952

- Spec-Augment
    Park, Daniel S., William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D. Cubuk, and Quoc V. Le. 
      "Specaugment: A simple data augmentation method for automatic speech recognition." in Interspeech (2019).
      doi: 10.21437/Interspeech.2019-2680
      
- Angular Additive Softmax (ArcFace)
    Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. 
      "Arcface: Additive angular margin loss for deep face recognition." in IEEE/CVF CVPR, (2019).
      doi: 10.1109/TPAMI.2021.3087709 from IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 44 (10), pp. 5962 - 5979, (2021).
      
[Code Reference]
- Official code implementation: [ https://github.com/nikvaessen/w2v2-speaker ]
    * FinetuneWV2: 
        https://github.com/nikvaessen/w2v2-speaker/blob/master/src/models/wav2vec2.py
    * BCE & CE Loss: 
        https://github.com/nikvaessen/w2v2-speaker/blob/master/src/optim/loss
    
- AngularAdditiveMarginSoftMaxLoss
    https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py

- Spec-Augment
    https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py

[SUMMARY]
- Overall architecture
    * The model class 'FinetuneWV2' comprises the Wav2Vec 2.0 backbone and the head layer respect to the model configuration 'variant'.
    * You can configure the head layer by giving 'CE', 'AAM' or 'BCE' value to '--variant'.
    * Note that, this code only reproduces the way of pooling method by 'first&cls' as introduced in (Sec. 3.3. Pooling methods & Sec. 4.5. Ablation study - Table 2.).
    * Unlike the original setup from N. Vaessan's git project, this reproduction implements a pad-irrelevant processing framework (attention masking on transformer layers), 
      which means that batch size doesn't have to be 1 at the evaluation phase.
    
- Spec-augment
    * Referring the paper, masking strategy is applied on the output of Convolutional Network within the training.
"""
import math
import numpy as np

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from torchaudio.models import wav2vec2_base, wav2vec2_large
from typing import Tuple, Optional

from einops import rearrange

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


# Code Reference: https://github.com/nikvaessen/w2v2-speaker/blob/master/src/optim/loss/binary_cross_entropy.py
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits: torch.Tensor, label_indexes: torch.Tensor):
        return self._bce_loss(logits, label_indexes)

    def _bce_loss(self, logits: torch.Tensor, label_indexes: torch.Tensor):
        # logits (unnormalized quantities on which sigmoid is applied)
        # with shape [BATCH_SIZE, 1] and
        # label indexes (integers in {0, 1}) with shape [BATCH SIZE]
        logits = logits.squeeze(-1).to(torch.float32)
        label_indexes = label_indexes.squeeze().to(torch.float32)

        loss = F.binary_cross_entropy_with_logits(logits, label_indexes)

        with torch.no_grad():
            # put predictions into [0, 1] range for later calculation of accuracy
            prediction = torch.sigmoid(logits).detach()

        return loss, prediction

class BCEHead(nn.Module):
    """ Wrapper module for the BinaryCrossEntropyLoss() with a linear projection head """
    def __init__(self, in_features):
        super(BCEHead, self).__init__()
        self.fc = nn.Linear(in_features, 1, bias=False)
        self.loss = BinaryCrossEntropyLoss()
    
    def forward(self, x:torch.Tensor, label:torch.Tensor=None):
        """
            x     (torch.Tensor.float) : [batch_size, embed_size]
            label (torch.Tensor.long)  : [batch_size] _binary values
        """
        x = self.fc(x)

        # training mode
        if label is not None:
            L, pred = self.loss(x, label)
            return pred, L
        
        # inference mode
        else:
            with torch.no_grad():
                pred = torch.sigmoid(x.squeeze(-1).to(torch.float32)).detach()
            return pred


# Code Reference: https://github.com/nikvaessen/w2v2-speaker/blob/master/src/optim/loss/cross_entropy.py
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, label_indexes: torch.Tensor):
        return self._ce_loss(logits, label_indexes)

    def _ce_loss(self, logits: torch.Tensor, label_indexes: torch.Tensor):
        # logits (unnormalized quantities on which softmax is applied)
        # with shape [BATCH_SIZE, NUM_SPEAKERS] and
        # label indexes (integers in range [0, NUM_SPEAKERS-1])
        # with shape [BATCH SIZE]
        loss = F.cross_entropy(logits, label_indexes)

        with torch.no_grad():
            # put predictions into [0, 1] range for later calculation of accuracy
            prediction = F.softmax(logits, dim=1).detach()

        return loss, prediction

class CEHead(nn.Module):
    """ Wrapper module for the CrossEntropyLoss() with a linear projection head """
    def __init__(self, in_features, out_features):
        super(CEHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.fc.weight)
        
        self.loss = CrossEntropyLoss()

    def forward(self, x:torch.Tensor, label:torch.Tensor):
        """
            x     (torch.Tensor.float) : [batch_size, embed_size]
            label (torch.Tensor.long)  : [batch_size] _categorical values
        """
        x = self.fc(x) # (B, Class)
        L, pred = self.loss(x, label)
        return pred, L


# Code Reference: 
#   https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
#   https://github.com/nikvaessen/w2v2-speaker/blob/master/src/optim/loss/aam_softmax.py
class AngularAdditiveMarginSoftMaxLoss(nn.Module):
    def __init__(
        self,
        input_features,
        output_features,
        margin=0.3,
        scale=15,
        easy_margin=False,
    ):
        super(AngularAdditiveMarginSoftMaxLoss, self).__init__()

        self.margin = margin
        self.scale = scale
        self.input_features = input_features
        self.fc_weights = torch.nn.Parameter(
            torch.FloatTensor(output_features, input_features), requires_grad=True
        )
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.fc_weights, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, x, label=None):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.input_features

        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.fc_weights))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        # one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        loss = self.ce(output, label)
        prediction = F.softmax(output, dim=1)

        # return loss, prediction
        return prediction, loss
    

# Code Reference : https://github.com/nikvaessen/w2v2-speaker/blob/master/src/models/wav2vec2.py
class SVmodel(nn.Module):
    """ Pytorch implementation of the proposed speaker verification model from the paper,
            "Fine-tuning wav2vec2 for speaker recognition." in IEEE ICASSP (2022).
            
    This reproduction adapts the original lightening framework into the Pytorch version.
    The implementation includes a pad-irrelevant processing framework (attention masking on transformer layers), 
    so that batch size doesn't have to be 1 at the evaluation phase.
    """
    def __init__(self, config:dict):
        super(SVmodel, self).__init__()
        
        self.config = config

        # Wav2Vec 2.0 -- base
        if config['model']['w2v_model_sz'] == 'large':
            self.embed_size = 1024
            if config['model']['use_pretrain']:
                self.w2v_model = torchaudio.pipelines.WAV2VEC2_LARGE.get_model()
            else:
                self.w2v_model = wav2vec2_large()

        elif config['model']['w2v_model_sz'] == 'base':
            self.embed_size = 768
            if config['model']['use_pretrain']:
                self.w2v_model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
            else:
                self.w2v_model = wav2vec2_base()

        # SpecAugment after the feature projection
        if self.config['model']['mask_time_prob'] > 0 or self.config['model']['mask_feature_prob'] > 0:
            self.specaug = True
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(self.embed_size).uniform_())

        # Training Head
        if config['model']['variant'] == 'BCE':
            self.BCE_mode = True
            self.training_head = BCEHead(self.embed_size)
        else:
            self.BCE_mode = False
            if config['model']['variant'] == 'CE':
                self.training_head = CEHead(self.embed_size, config['data']['nb_class_train'])
                
            elif config['model']['variant'] == 'AAM':
                # use scale 30., and margin 0.2 (ref. to Sec. 4.3.)
                # self.training_head = AAMHead(self.embed_size, config['data']['nb_class_train'], s=30.0, m=0.2)
                self.training_head = AngularAdditiveMarginSoftMaxLoss(input_features=self.embed_size,
                                                                      output_features=config['data']['nb_class_train'],
                                                                      scale=30.0,
                                                                      margin=0.2)
            else:
                raise NotImplementedError(config['model']['variant'])
    
    # Code Reference : https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1478
    def _spec_augment(self, hidden_states:torch.Tensor):
        """ Mask extracted features along time axis and/or along feature axis
        
        Input:
            hidden_states (torch.Tensor.float) : [batch_size, max_lenth(time), embedding_size(feature)]
        """
        
        if self.specaug:
            B, T, E = hidden_states.size()
            
            if self.config['model']['mask_time_prob'] > 0: # Time-axis Augmentation
                mask_indices = _compute_mask_indices(
                    shape = (B, T),
                    mask_prob=self.config['model']['mask_time_prob'],
                    mask_length=self.config['model']['mask_time_length']
                    )
                mask_indices = torch.tensor(mask_indices, device=hidden_states.device, dtype=torch.bool)
                hidden_states[mask_indices] = self.masked_spec_embed.to(hidden_states.dtype)
            
            if self.config['model']['mask_feature_prob'] > 0: # Feature-axis Augmentation
                mask_indices = _compute_mask_indices(
                    shape=(B, E),
                    mask_prob=self.config['model']['mask_feature_prob'],
                    mask_length=self.config['model']['mask_feature_length']
                    )
                mask_indices = torch.tensor(mask_indices, device=hidden_states.device, dtype=torch.bool)
                mask_indices = mask_indices[:, None].expand(-1, T, -1)
                hidden_states[mask_indices] = 0
            
        else:
            pass
        
        return hidden_states
    
    def _insert_tokens_training_bce(self, features, lengths):
        """
            features: (B, T, E)
            lengths:  (B)
        """
        
        features = rearrange(features, "(b p) T E -> b p T E", p=2)
        b, p, t, E = features.size()
        
        # Split the pairs        
        f0, f1 = torch.split(features, 1, dim=1)
        f0 = f0.squeeze(1); l0 = lengths[0::2].to(int)
        f1 = f1.squeeze(1); l1 = lengths[1::2].to(int) # f0/1 : (b, T, E), l0/1 : (b)

        # Insert CLS(+1) token at the beginning
        T = 2 * t + 3
        sq = torch.cat([torch.ones(size=(b, 1, E), device=features.device), f0], dim=1)
        sq = F.pad(sq, (0,0,0,T-(t+1)), 'constant', 0) # sequence: (b, 2*t+3, E)
        l0 = l0 + 1
        
        # SEP(-1) at the middle
        sq[:, l0] = -1
        l0 = l0 + 1
        
        # second utterance with SEP(-1) at the end
        f1 = F.pad(f1, (0,0,0,1), 'replicate')
        f1[:, l1] = -1
        l1 = l1 + 1
        
        # concatenate two utterances
        for i, v in enumerate(f1):
            sq[i, l0[i]:l0[i]+l1[i]] = v[:l1[i]]
        outlen = l0 + l1
        
        return sq[:, :outlen.max().item()], outlen
    
    def _insert_tokens_ce(self, features, lengths=None):
        """
            features: (B, T, E)
            lengths:  (B)
        """
        B, T, E = features.size()
        
        # Insert CLS(+1) token at the beginning
        sq = torch.cat([torch.ones(size=(B, 1, E), device=features.device), features], dim=1) # sq: (B, T+1, E)
        lengths = (lengths + 1).to(int) if lengths is not None else None
        
        return sq, lengths
        
    def forward(self, waveforms, lengths=None, targets=None):
        """
        Input:
            waveforms (torch.Tensor.float) : [batch_size, max_length]
            lengths   (torch.Tensor.long)  : [batch_size] _lengths for each audio sample in the batch
            targets   (torch.Tensor.long)  : [batch_size] _categorical | _binary values
            
        Output:
            if train:
                logit (torch.Tensor.float)
                    if BCE : 
                        [batch_size, 2] _final output = torch.stack([1.-x, x], dim=-1), where x is sigmoid() output
                    if CE | AAM : 
                        [batch_size, nb_classes] _logits
                L (torch.Tensor.float) : [1] 
                
            else inference:
                outputs (torch.Tensor.float) : [batch_size] _cosine similarity | _probability to be the same
        """
        if self.training:
            
            # W2V2 Conv-layer & feature projection
            features, lengths = self.w2v_model.feature_extractor(waveforms, lengths) # (B, T, 512)
            features = self.w2v_model.encoder.feature_projection(features) # (B, T, 768/1024)
            
            # SpecAugment
            features = self._spec_augment(features)
            
            # Insert special tokens
            if self.BCE_mode:
                sequence, lengths = self._insert_tokens_training_bce(features, lengths)   
            else: # CE | AAM loss head
                sequence, lengths = self._insert_tokens_ce(features, lengths)
                
            # TF-layers
            B, T, E = sequence.size()
            mask = (torch.arange(T, device=features.device) < lengths[:, None]).unsqueeze(-1).repeat(1, 1, T)
            mask = (mask & mask.transpose(1,2)).unsqueeze(1)      # mask : (B, 1, T, T)
            outputs = self.w2v_model.encoder.transformer(sequence, mask) # outputs: (B, T, E)
            
            # Pool embedding from the CLS position
            outputs = outputs[:, 0] # outputs: (B, E)
            
            return self.training_head(outputs, targets)
        
        else: # self.eval
            
            if self.BCE_mode:
                # Conv-layers (waveforms: (B=2, T)) & projection
                f0, _ = self.w2v_model.feature_extractor(waveforms[0], lengths)
                f1, _ = self.w2v_model.feature_extractor(waveforms[1], lengths)
                f0    = self.w2v_model.encoder.feature_projection(f0) # (B=1, T, 768/1024)
                f1    = self.w2v_model.encoder.feature_projection(f1) # (B=1, T, 768/1024)
                E = f0.size(-1)
                
                # Insert Special Tokens
                sequence = torch.cat([torch.ones(size=(1, 1, E), device=f0.device),
                                      f0,
                                      torch.ones(size=(1, 1, E), device=f0.device) * -1,
                                      f1,
                                      torch.ones(size=(1, 1, E), device=f0.device) * -1], dim=1)
                
                # TF-layers
                outputs = self.w2v_model.encoder.transformer(sequence)
                outputs = outputs[:, 0] # outputs: (B=1, E)
                
                return self.training_head(outputs) # return [0-1] similarity
            
            else: # CE | AAM loss head
                # Conv-layers / projection & Insert Special Token
                features, _ = self.w2v_model.feature_extractor(waveforms, lengths) # features (B=1, T, 512)
                features    = self.w2v_model.encoder.feature_projection(features) # (B=1, T, 768/1024)
                sequence, _ = self._insert_tokens_ce(features)
                # TF-layers
                outputs = self.w2v_model.encoder.transformer(sequence)
                outputs = outputs[:, 0] # outputs: (B=1, E)
                
                return F.normalize(outputs, dim=-1) # return (B=1, E) embedding
                
        
