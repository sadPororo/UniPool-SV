""" This is a model reproduction of proposal from the paper, "ECAPA-TDNN: Emphasized Channel Attention, propagation and aggregation in TDNN based speaker verification."

[Paper Reference]
- Speaker verification/identification model
    Brecht Desplanques and Jenthe Thienpondt and Kris Demuynck.
      "ECAPA-TDNN: Emphasized Channel Attention, propagation and aggregation in TDNN based speaker verification." in Interspeech (2020).
      doi: 10.1109/ICASSP43922.2022.9746952
    
- Angular Additive Softmax (ArcFace)
    Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. 
      "Arcface: Additive angular margin loss for deep face recognition." in IEEE/CVF CVPR, (2019).
      doi: 10.1109/TPAMI.2021.3087709 from IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 44 (10), pp. 5962 - 5979, (2021).
      
[Code Reference]
- Community code implementation: [ https://github.com/TaoRuijie/ECAPA-TDNN/tree/main ]
    * ECAOA_TDNN architecture:
        https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/model.py
    * AAMsoftmax loss:
        https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/loss.py

[Summary]
- ECAPA_TDNN_Wrapper
    * This wrapper class comprises "ECAPA_TDNN" backbone architecture and "AAMsoftmax" loss head.

- ECAPA_TDNN
    * There are some modifications applied to be more close to the configuration specified at the original paper.
      1) SpecAugment Configuration
          I. Adjust "class FbankAug::__init__()::freq_mask_width=(0,8) & time_mask_width=(0,10)" to "freq_mask_width=(0, 10) & time_mask_width=(0,5)".
      2) Input Audio Feature Extraction
          I. Replace "torchaudio.transforms.MelSpectrogram()" from "class ECAPA_TDNN::__init__()::self.torchfbank" into "torchaudio.transforms.MFCC()"
         II. Therefore, the feature extraction part from "class ECAPA_TDNN::forward()" is also adjusted.
            
- AAMsoftmax
    "class AAMsoftmax::forward()" is modified to output probability and loss, from producing accuracy and loss.
"""
# Code Reference: https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/model.py
#                 https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/loss.py
# Every class/function below are taken from the code implementations above. (except the wrapper class 'ECAPA_TDNN_Wrapper')
'''
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py

'''
#%%
import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 

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

class FbankAug(nn.Module):

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


class ECAPA_TDNN(nn.Module):

    def __init__(self, C):

        super(ECAPA_TDNN, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            # [Original Code Implementation]----------------------------------------------------------------------------------
            # torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
            #                                      f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            # [Modification]--------------------------------------------------------------------------------------------------
            torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=80, log_mels=False,
                                        melkwargs={'n_fft': 512, 'win_length':400, 'hop_length':160, 
                                                    'f_min':20, 'f_max':7600, 'window_fn': torch.hamming_window, 'n_mels':80})
            # ----------------------------------------------------------------------------------------------------------------
            )

        self.specaug = FbankAug() # Spec augmentation

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper. 
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug):
        with torch.no_grad():
            # [Original Code Implementation]---------------------
            # x = self.torchfbank(x)+1e-6
            # x = x.log()   
            # x = x - torch.mean(x, dim=-1, keepdim=True)
            # [Modification]-------------------------------------
            x = self.torchfbank(x)    # (batch, n_mfcc, t_frames)
            x = F.normalize(x, dim=1)
            #----------------------------------------------------
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x
    

# Code Reference: https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/loss.py
#    - Copied from: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
#       - Originally adapted from: https://github.com/wujiyang/Face_Pytorch (Apache License)
class AAMsoftmax(nn.Module):
    """An implementation of Additive Angular Margin Loss
    
    method::forward() originally outputs loss and accuracy of the model prediction, however, this is modified to output probability and loss.
    Therefore accumulation and scoring of the model prediction proceed outside the model class, done by the function::evaluate() at trainer.py.
    """
    def __init__(self, n_class, m, s):
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):        
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        loss = self.ce(output, label)
        # [Original Code Implementation]------------------------------------
        # prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        # return loss, prec1
        #
        # [Modification]----------------------------------------------------
        return output, loss
        # ------------------------------------------------------------------


########################################################################################################################################################
# Code Reference: https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/ECAPAModel.py
class SVmodel(nn.Module):
    """ The class implements the data flow in training/inference phase of ECAPA_TDNN proposed from the paper,
            "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker." in Interspeech (2020).
    """    
    def __init__(self, config:dict):
        super(SVmodel,self).__init__()
        self.speaker_encoder = ECAPA_TDNN(C=config['model']['n_channels'])
        self.aamsoftmax_loss = AAMsoftmax(n_class=config['data']['nb_class_train'], m=0.2, s=30)
    
    def forward(self, waveform, target=None):
        """
        Args:
            waveforms (torch.Tensor.float) : [batch_size, max_length]
            targets   (torch.Tensor.long)  : [batch_size] _categorical values
            
        Output:
            if train:
                logit (torch.Tensor.float) : [batch_size, nb_classes] _cosine similarity
                L     (torch.Tensor.float) : [1] _additive margin softmax loss
                
            else inference:
                output (torch.Tensor.float) : [batch_size, embed_size=192]
        """
        if target is not None: # on training
            x = self.speaker_encoder(waveform, aug=True)
            c_prob, L = self.aamsoftmax_loss(x, label=target)
            return c_prob, L
        
        else: # inference
            return F.normalize(self.speaker_encoder(waveform, aug=False), p=2, dim=-1)        
            

########################################################################################################################################################
# if __name__ == "__main__":
#     import yaml
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import librosa
    
#     config = {
#         'data':{
#             'nb_class_train':72
#             },
        
#         'model': {
#             'n_channels': 512,
#         }
#     }

#     device = torch.device('cuda:0')    
#     model = ECAPA_TDNN_Wrapper(config).to(device)
#     sr = 16000
    
#     batch_size = 8
#     train_max_sec = 2
#     wlen = int(sr*train_max_sec)
#     num_eval_sample = 10
    
#     # training sample ##########################################
#     print('- train')
#     waveform = ((torch.rand(batch_size, int(wlen)) -.5) * 2)   # 16000 sample rate
#     target   = torch.randint(low=0, high=config['data']['nb_class_train'], size=(waveform.size(0),))
    
#     waveform = waveform.to(device); target = target.to(device)
#     optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    
#     model.train()
#     prob, loss = model(waveform, target)
#     pred = prob.detach().cpu().argmax(dim=-1)
#     print(pred.long())
#     print(target.long())
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     print(loss.item())
    
#     # eval ####################################################
#     print('- eval')
#     model.eval()
    
#     length = ((torch.ones(2) * wlen) + (torch.rand(2) + 1.).softmax(dim=-1) * wlen * torch.randint(low=-1, high=4, size=(2,))).long()    
#     waveform = ((torch.rand(2, length.max().item()) -.5) * 2)   # 16000 sample rate
#     target = [0]

#     plt.bar(range(len(length)), length.numpy())
#     plt.plot(torch.ones_like(length).numpy() * wlen, color='red')
#     plt.show()

#     with torch.no_grad():
                
#         embedding_list = []
#         for i in range(length.size(0)):
#             lt = length[i].item()
#             wave = waveform[i, :length[i]] # (1, T)
            
#             plt.figure(figsize=(int(len(wave) / sr), 5))
#             librosa.display.specshow(model.speaker_encoder.torchfbank(wave.unsqueeze(0).to(device)).squeeze(0).detach().cpu().numpy(), 
#                                      x_axis='time', sr=sr, hop_length=int(0.01*sr), fmin=20, fmax=7600)
#             plt.show();
            
#             if lt <= wlen: # padding
#                 shortage = wlen - lt
#                 wave = torch.from_numpy(np.pad(wave, (0, shortage), mode='wrap')).unsqueeze(0) # (1, 6000)
#                 # wave = F.pad(wave.unsqueeze(0), (0,shortage), mode='circular').squeeze(0) # (1, 6000)
            
#             else: # sampling
#                 chunked = []; wave = wave.squeeze(0)
#                 for idx in np.linspace(0, lt-wlen, num=num_eval_sample):
#                     chunked.append(wave[int(idx):int(idx)+wlen])
#                 wave = torch.stack(chunked)
    
#             output = model(wave.to(device))
#             embedding_list.append(output)
            
#             print(output.size())
        
#         # sim_cdist = torch.cdist(embedding_list[0], embedding_list[1]).detach().cpu().numpy() * -.5 + 1.
#         sim_matml = (torch.matmul(embedding_list[0], embedding_list[1].T)).detach().cpu().numpy() *.5 + .5
        
#         # print(sim_cdist)
#         print(sim_matml)
#         # print(np.mean(sim_cdist))
#         print(np.mean(sim_matml))

#     # # get_EER(torch.Tensor(targets), outputs)


# %%

