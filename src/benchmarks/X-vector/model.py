""" The implementation of X-vector Speaker verification system from the paper, "X-vectors: Robust DNN Embeddings for Speaker Recognition." in IEEE ICASSP (2018).

[Paper Reference]
- Speaker verification/identification model
    David Snyder, Daniel Garcia-Romero, Gregory Sell, Daniel Povey, Sanjeev Khudanpur. 
      "X-vectors: Robust DNN Embeddings for Speaker Recognition." in IEEE ICASSP (2018).
      doi: 10.1109/ICASSP43922.2022.9746952
      
[Code Reference]
- Community code implementation: [ https://github.com/cvqluu/TDNN ]
    * TDNN:
        https://github.com/cvqluu/TDNN

- MFCC configurations are referred from: 
    https://github.com/kaldi-asr/kaldi/tree/master/egs/sre16/v2
- PreEmphasis: 
    https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/model.py/#L82
    
[Summary]
- X_vector
    * backbone model to embed fixed size speaker representation from mfcc spectrogram

- X_vector_Wrapper
    * The wrapper class including whole architecture comprising audio-feature-extraction, model, and loss-head.
    * Some modifications are applied to be more close to the configuration specified from the original paper.
    1) Feature Extraction:
        I. We followed the number of mfcc frequency bins as specified in the paper. (24 bins)
       II. 'f_max' argument is set to 7600 due to the 16k sample rate (Nyquist), while the argument 'high-freq' is set to 3700 with 8k sample rate from the kaldi-asr toolkit.
"""
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# Code Reference: https://github.com/cvqluu/TDNN/blob/master/tdnn.py
class TDNN(nn.Module):
    
    def __init__(
                    self, 
                    input_dim=23, 
                    output_dim=512,
                    context_size=5,
                    stride=1,
                    dilation=1,
                    batch_norm=True,
                    dropout_p=0.0
                ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

        Affine transformation not applied globally to all frames but smaller windows with local context

        batch_norm: True to include batch normalisation after the non linearity
        
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
      
        self.kernel = nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
                        x, 
                        (self.context_size, self.input_dim), 
                        stride=(1,self.input_dim), 
                        dilation=(self.dilation,1)
                    )

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1,2)
        x = self.kernel(x)
        x = self.nonlinearity(x)
        
        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1,2)
            x = self.bn(x)
            x = x.transpose(1,2)

        return x


# Code Reference: https://github.com/cvqluu/TDNN
class X_vector(nn.Module):
    """ Implementaion of the x-vector system proposed from the paper reference. """
    def __init__(self, input_dim=24):
        super(X_vector, self).__init__()
                
        # Frame-level (frame1 - 5)
        self.frame_level_layers = nn.Sequential(
            TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1),  # Input to frame1 is of shape (batch_size, T, 24)
            TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2),
            TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3),
            TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1),
            TDNN(input_dim=512, output_dim=1500, context_size=1, dilation=1)        # Output of frame5 will be (batch_size, T-14, 1500)
        ) # Followed by Stats Pooling -- torch.cat([mean, std], dim=-1) -> (batch_size, 3000)
        
        # Segment-level (segment6, 7)
        self.segment6 = nn.Linear(3000, 512) # -> x_vector
        self.act6 = nn.ReLU()
        self.bn6  = nn.BatchNorm1d(num_features=512)
        
        self.segment7 = nn.Linear(512, 512)
        self.act7 = nn.ReLU()
        self.bn7  = nn.BatchNorm1d(num_features=512)
        
        self.out_dim  = 512
    
    def forward(self, mfccs, length=None, is_eval=False):
        """
        Args:
            mfccs (torch.Tensor): [batch_size, T, 24], T=3 second for training
            length (torch.LongTensor): [batch_size,]

        Returns:
            x_vector: [batch_size, 512]
        """
        # TDNN layers (frame-level)
        x = self.frame_level_layers(mfccs)
        B, T, _ = x.size()
        
        # Statistical Pooling
        if length is not None:
            length = length - 14
            length[length<=0] = 1
            
            mask   = (torch.arange(T, device=x.device) < length[:, None]).unsqueeze(-1) # (B, T, 1)
            x_mean = (x * mask).sum(dim=1) / length.unsqueeze(-1) # (B, C=1500)
            x_std  = torch.sqrt((((x - x_mean.unsqueeze(1)) ** 2) * mask).sum(dim=1) / length.unsqueeze(-1)) # (B, C=1500)
        else:
            x_mean = x.mean(dim=1)
            x_std  = x.std(dim=1)

        x = torch.cat([x_mean, x_std], dim=-1) # (B, 3000)
        
        # FC layers (segment-level)
        x = self.segment6(x)
        if is_eval:
            return x # x-vector
        x = self.bn6(self.act6(x))
        
        x = self.bn7(self.act7(self.segment7(x)))
        return x
        

class CEHead(nn.Module):
    def __init__(self, in_features, out_features):
        super(CEHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.fc.weight)
        
    def forward(self, x:torch.Tensor, label:torch.Tensor):
        """
            x     (torch.Tensor.float) : [batch_size, embed_size]
            label (torch.Tensor.long)  : [batch_size] _categorical values
        """
        logits = self.fc(x) # (B, Class)
        loss = F.cross_entropy(logits, label)
        
        with torch.no_grad():
            pred = F.softmax(logits, dim=1).detach()
            
        return pred, loss


# Code Reference: https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/model.py/#L82
class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


# Code Reference: https://github.com/TaoRuijie/ECAPA-TDNN/tree/main
class FbankAug(nn.Module):
    """ Custom Implementation of SpecAugment
    
    [Paper Reference]
        Park, Daniel S., William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D. Cubuk, and Quoc V. Le. 
        "Specaugment: A simple data augmentation method for automatic speech recognition." in Interspeech (2019).
        doi: 10.21437/Interspeech.2019-2680
    """
    # [Original Code Implementation]------------------------------------------
    # def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
    #
    # [Modification]----------------------------------------------------------
    def __init__(self, freq_mask_width = (0, 10), time_mask_width = (0, 5)):
        # --------------------------------------------------------------------
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x



class SVmodel(nn.Module):
    """ Wrapper class for the x-vector system

    - feature_extraction: extract MFCCs from raw-waveform
        * Here, we use f_max=7600 due to the difference of the sample rate from the reference (sr=8000). (Nyquist Frequency)
    - speaker_encoder: the backbone model with TDNN-based layers, statistical pooling and fully connected layers, which produces x-vector.
    - loss_head: cross entropy loss
    """
    def __init__(self, config:dict):
        super(SVmodel, self).__init__()
        self.hop_length = 160  # 10ms on 16k sample rate
        self.eps = 1e-8
        self.feature_extraction = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=24, log_mels=False,
                                        melkwargs={'n_fft': 512, 'win_length':400, 'hop_length':160, 
                                                   'f_min':20, 'f_max':7600, 'window_fn': torch.hamming_window, 'n_mels':80})
            )
        if config.get('model') is not None:
            self.withaug = config['model']['specaug']
        else:
            self.withaug = False
        # self.withaug = config['model']['specaug']
        self.specaug = FbankAug() # Spec augmentation
        
        self.speaker_encoder = X_vector(input_dim=24)
        self.loss_head = CEHead(in_features=self.speaker_encoder.out_dim,
                                out_features=config['data']['nb_class_train'])
        
    def forward(self, waveform:torch.Tensor, length=None, target=None):
        """
        Args:
            x (torch.Tensor): [batch_size, max_length]
            length (torch.LongTensor, optional): [batch_size]. Defaults to None.
            target (torch.LongTensor, optional): [batch_size]. Defaults to None.
        """
        # Feature Extraction
        with torch.no_grad():
            mfccs = self.feature_extraction(waveform) # (B, mfcc, T)
            mfccs = mfccs.transpose(1, 2) # (B, T, mfccs)
            
            # mean normalization
            if length is not None:                
                length = torch.floor(length/self.hop_length + 1).long()
                
                _, T, _ = mfccs.size()
                mask = (torch.arange(T, device=mfccs.device) < length[:, None]).unsqueeze(-1) # (B, T, 1)
                mu   = ((mfccs * mask).sum(dim=1) / length.unsqueeze(-1)).unsqueeze(1) # (B, 1, mfcc)
                sig  = (torch.sqrt((((mfccs - mu) ** 2) * mask).sum(dim=1) / length.unsqueeze(-1))).unsqueeze(1) # (B, 1, mfcc)
            else:
                mu  = mfccs.mean(dim=1, keepdim=True) # (B, 1, mfcc)
                sig = mfccs.std(dim=1, keepdim=True)  # (B, 1, mfcc)
            
            sig[sig < self.eps] = self.eps
            mfccs = (mfccs - mu) / sig
            
            # # SpecAugment
            # if self.withaug and target is not None:
            #     mfccs = self.specaug(mfccs)
            
        # return x-vector (on inference) or classification prediction with loss (on training)
        if target is not None:
            # get X-vector
            x = self.speaker_encoder(mfccs, length)
            pred, L = self.loss_head(x, target)
            return pred, L
        else:
            # get X-vector
            x = self.speaker_encoder(mfccs, length, is_eval=True)
            return F.normalize(x, dim=-1)
            
