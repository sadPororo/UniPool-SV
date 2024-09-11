""" Dataset utilities/functions in common """
import os
import random
import textgrids
import torch
import torchaudio

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.nn.functional import pad

from os.path import join as opj
from itertools import cycle, combinations


def pad_collate_multiclass(batch):
    """ For the task of classification among the multiple-classified audio samples.
    
    Args:
        batch (list of tuples): [(waveform, label)]
            waveform (FloatTensor)
            label    (LongTensor)

    Returns:
        waveforms (FloatTensor) : [batch_size, max_length]
            * max_length : maximum audio length in the batch
        lengths   (LongTensor)  : [batch_size]
        labels    (LongTensor)  : [batch_size]
    """
    if len(batch) == 1:
        waveforms = batch[0][0].unsqueeze_(0) # (1, wave_length)
        lengths   = torch.LongTensor([len(batch[0][0])]) # (1)
        labels    = torch.LongTensor([batch[0][1]]) # (1)
    
    if len(batch) > 1:
        waveforms, labels, lengths = zip(*[(a, b, len(a)) for (a, b) in batch])
        max_length = max(lengths)
        waveforms  = [pad(w, (0, max_length - lengths[i]), 'constant', 0) for i, w in enumerate(waveforms)]
        
        waveforms = torch.stack(waveforms, dim=0)
        lengths   = torch.LongTensor(lengths)
        labels    = torch.LongTensor(labels)
        
    return waveforms, lengths, labels


def pad_collate_binary(batch):
    """ For the task of binary decision (Match | Unmatch) within two audio samples.
     
    Args:
        batch (list of tuples): [(waveform0, waveform1, label)]
            waveform0 (FloatTensor)
            waveform1 (FloatTensor)
            label    (LongTensor)

    Returns:
        waveforms (FloatTensor) : [batch_size, 2, max_length]
            * max_length : maximum audio length in the batch
        lengths   (LongTensor)  : [batch_size, 2]
        labels    (LongTensor)  : [batch_size]
    """
    if len(batch) == 1:
        wave0 = batch[0][0] # (wave0_length,)
        wave1 = batch[0][1] # (wave1_length,)

        if len(wave0) >= len(wave1):
            wave1 = pad(wave1, (0, len(wave0) - len(wave1)), 'constant', 0)
        else:
            wave0 = pad(wave0, (0, len(wave1) - len(wave0)), 'constant', 0)
            
        waveforms = torch.stack([wave0, wave1], dim=0).unsqueeze_(0) # (1, 2, max_length)
        lengths   = torch.LongTensor([len(wave0), len(wave1)]).unsqueeze_(0) # (1, 2)
        labels    = torch.LongTensor([batch[0][2]]) # (1)
        
    if len(batch) > 1:
        wave0, wave1, len0, len1, labels = zip(*[(a, b, len(a), len(b), c) for (a, b, c) in batch])
        max_length = max([max(len0), max(len1)])
        waveforms  = [torch.stack([pad(w0, (0, max_length - len0[i]), 'constant', 0), 
                                   pad(w1, (0, max_length - len1[i]), 'constant', 0)], dim=0) for i, (w0, w1) in enumerate(zip(wave0, wave1))]
        lengths    = [torch.LongTensor([a, b]) for a, b in zip(len0, len1)]

        waveforms = torch.stack(waveforms, dim=0)
        lengths   = torch.stack(lengths, dim=0)
        labels    = torch.Tensor(labels)
        
    return waveforms, lengths, labels


def crop_audio(waveform:torch.Tensor, filepath:str, max_second:int, max_length:int, sr:int, aligned_crop:bool=False):
    """ Crop audio sample into the maximum training length
    
    Args:
        waveform (torch.Tensor): sample waveform
        filepath (str): file path of given waveform
        max_second (int): the maximum length of training audio in seconds
        max_length (int): the maximum length of training audio in Tensor length
        sr (int): sample rate of given audio waveform
        aligned_crop (bool): when True, crop the wave sample on the word alignments if the sample comprises the transcript (TextGrid) and audio.
    
    Returns:
        waveform (torch.Tensor): cropped into the length the same or less than the train-maximum
    """
    
    if aligned_crop and os.path.isfile(filepath.replace('.wav', '.TextGrid')):
        # read .TextGrid file
        txtgrid = textgrids.TextGrid(filepath.replace('.wav', '.TextGrid')).interval_tier_to_array('words')
        threshold = txtgrid[-1]['end'] - max_second
        start_idx = [i for (i, word) in enumerate(txtgrid) if word['begin'] <= threshold and word['label'] != '']
        start_idx = random.choice(start_idx)
        
        _, sample_start, sample_end = list(txtgrid[start_idx].values())
        sample_sec = 0
        while True:
            label, word_begin, word_end = list(txtgrid[start_idx].values())
            word_length = word_end - word_begin
            if sample_sec + word_length > max_second:
                break
            sample_end = word_end
            sample_sec += word_length
            start_idx  += 1
            if start_idx >= len(txtgrid):
                break
            
        waveform = waveform[int(sample_start*sr):int(sample_end*sr)]
        
    else:
        start_idx = random.choice(list(range(len(waveform)-max_length+1)))
        waveform = waveform[start_idx:start_idx+max_length]
        
    return waveform


def pad_audio(waveform:torch.Tensor, max_length:int, mode='wrap'):
    """ Pad audio sample into the maximum training length
    
    Args:
        waveform (torch.Tensor): sample waveform
        max_length (int): the maximum length of training audio in Tensor length
        mode (str): refers to the "numpy.pad()" method argument
    
    Returns:
        waveform (torch.Tensor): padded to the train-maximum length    
    """
    return torch.from_numpy(np.pad(waveform, (0, max_length - len(waveform)), mode=mode))


def gather_train_filepaths(config:dict):
    """ Gather the whole file paths for configured training set.

    Args:
        config (dict): hyperparameters

    Returns:
        speaker_list (list): List of speaker-ID (str)
        sample_dict  (dict): Dictionary of speaker ID to the list of training samples 
                                                ... { speaker-id: list[ Tuple (label (int) , filepath (str) ), ...], ...}
        label_2_id   (dict): Dictionary of speaker-ID to class index (int)     ... { speaker-id : categorical index, ...}
        id_2_label   (dict): Dictionary of class index to speaker-ID           ... { categorical index : speaker-id, ...}
    """    
    
    dataset = config['posarg']['data'] # (e.g.) LibriSpeech | VCTK | Vox1 | Vox2 | ...
    data_path = config['data']['data_path']
    
    if dataset in ['VCTK', 'LibriSpeech']:
        meta_suffix =  'speaker-train.csv'
    elif 'Vox1' in dataset:
        meta_suffix =  'speaker-vox1-dev.csv'
    elif 'Vox2' in dataset:
        meta_suffix =  'speaker-vox2-dev.csv'
    else:
        raise NotImplementedError(dataset)
    
    # read training speaker meta file
    speaker_meta = pd.read_csv(opj(config['data']['speaker_meta'], meta_suffix))
    if speaker_meta['ID'].dtypes != str:
        speaker_meta['ID'] = speaker_meta['ID'].astype(str)
        
    speaker_list = speaker_meta['ID'].astype(str).to_list()
    
    # speaker-id to class index conversion, and vice versa. (for Multiclass Classification Training)
    label_2_id = speaker_meta['ID'].to_dict()
    id_2_label = {v: k for k, v in label_2_id.items()}
    
    # gather (label, filepath) for training set.
    sample_dict = {}
    if dataset=='VCTK':
        for s_id in speaker_list: # p225, p226, ... p376
            sample_dict[s_id] = []
            for f_name in os.listdir(opj(data_path, s_id)):
                if f_name.endswith('.wav'):
                    w_id = int(f_name[:-4].split('_')[-1]) # 001.wav, 002.wav, ...
                    if w_id < 25: 
                        continue # exclude the common scripts: no.001 ~ 024 >> (e.g.) 001.wav: "Please call Stella"
                    sample_dict[s_id].append((id_2_label[s_id], opj(s_id, f_name)))
                    
    elif dataset=='LibriSpeech':
        subset_list  = config['data']['train_subsets'] # ['train-clean-100', 'train-clean-360', 'train-others-500']
        subset_meta  = speaker_meta[['ID', 'SUBSET']].set_index('ID').T.to_dict('records')[0] # {'14': 'train-clean-360, '19': 'train-clean-100', ...}
        for s_id in speaker_list: # 14, 16, 17, 19, ...
            subset = subset_meta[s_id]
            if subset not in subset_list:
                continue
            sample_dict[s_id] = []
            
            for chap_no in os.listdir(opj(data_path, subset, s_id)):
                for f_name in os.listdir(opj(data_path, subset, s_id, chap_no)):
                    
                    if f_name.endswith('.wav'):
                        sample_dict[s_id].append((id_2_label[s_id], opj(subset, s_id, chap_no, f_name)))
                    
    elif 'Vox1' in dataset:
        subset = 'vox1_dev_wav'
        for s_id in speaker_list: # id10001, id10002, id10003, ...
            sample_dict[s_id] = []
            for v_id in os.listdir(opj(data_path, subset, s_id)): # 1zcIwhmdeo4, 7gWzIy6yIIk, 7w0IBEWc9Qw, ...
                for f_id in os.listdir(opj(data_path, subset, s_id, v_id)): # 00001.wav, 00002.wav, ...
                    sample_dict[s_id].append((id_2_label[s_id], opj(subset, s_id, v_id, f_id)))

    elif 'Vox2' in dataset:
        subset = 'vox2_dev_wav'
        for s_id in speaker_list: # id00012, id10015, id10016, ...
            sample_dict[s_id] = []
            for v_id in os.listdir(opj(data_path, subset, s_id)): # _raOc3-IRsw, 2DLq_Kkc1r8, 21Uxsk56VDQ, ...
                for f_id in os.listdir(opj(data_path, subset, s_id, v_id)): # 00110.wav, 00111.wav, ...
                    sample_dict[s_id].append((id_2_label[s_id], opj(subset, s_id, v_id, f_id)))

    else:
        raise NotImplementedError(dataset)
    
    return speaker_list, sample_dict, label_2_id, id_2_label


class MulticlassTrainDataset(Dataset):
    """ Training Dataset for the Multiclass Classification task
    
    Args:
        config (dict): hyperparameters
        
    Attributes:
        data_path (str) : path to the root data folder.
        nb_sample_for_iter (int) : the number of samples per training iteration.
        label_2_id (dict) : a dictionary of speaker ID (str) to class index (int)     ... { speaker-id : categorical index, ...}
        id_2_label (dict) : a dictionary of class index (int) to speaker ID (str)     ... { categorical index : speaker-id, ...}
        sample_list (list) : a list of training samples (label (int), filepath (str)) ... [ Tuple (label , filepath), ...]
        sample_cycle (cycle): an iterable of training samples
        train_samples (list): a sampled list of training instances from the "sample_list" for a single training iteration.
        
    * see 'pad_collate_multiclass()' for the detailed mini batch output shape.
    """
    def __init__(self, config:dict):        
        self.sr         = config['common']['sr']
        self.data_path  = config['data']['data_path']
        self.max_second = config['common']['train_max_sec']
        self.max_length = int(config['common']['sr'] * config['common']['train_max_sec'])
        self.fix_to_maxlen = config['common']['fix_to_maxlen']
        self.nb_sample_for_iter = config['common']['nb_steps_eval'] * config['common']['batch_size']
        
        _, sample_dict, self.label_2_id, self.id_2_label = gather_train_filepaths(config)
        self.sample_list = []
        for v in sample_dict.values():
            self.sample_list.extend(v)
        random.shuffle(self.sample_list)
        
        self.sample_cycle = cycle(self.sample_list)
        self.sample_for_iteration()
        
    def sample_for_iteration(self):
        self.train_samples = []
        for _ in range(self.nb_sample_for_iter):
            self.train_samples.append(next(self.sample_cycle))
            
    def get_total_dataset_size(self):
        return len(self.sample_list)
        
    def get_id_from_label(self, s_id):
        return self.id_2_label[s_id]
    
    def get_label_from_id(self, label):
        return self.label_2_id[label]
            
    def __len__(self):
        return len(self.train_samples)
    
    def __getitem__(self, index):
        label, filepath = self.train_samples[index]

        waveform, _ = torchaudio.load(opj(self.data_path, filepath))
        waveform = waveform.squeeze(0)
        
        if len(waveform) > self.max_length:
            waveform = crop_audio(waveform, opj(self.data_path, filepath), self.max_second, self.max_length, self.sr)
        elif self.fix_to_maxlen:
                waveform = pad_audio(waveform, self.max_length)

        return waveform, label
    
    
class PairedTrainDataset(Dataset):
    """ Training Dataset for the Binary classification | Contrastive learning from the paired audio samples.
    
    Args:
        config (dict): hyperparameters
        
    Attributes:
        data_path (str) : path to the root data folder.
        nb_sample_for_iter (int) : the number of samples per training iteration.
        speaker_list (list) : a list of speaker-ID (str)
        pos_speaker_cycle (cycle): an iterable of speaker-IDs
        neg_speaker_cycle (cycle): an iterable of the paired speaker-ID combinations
        
    * see 'pad_collate_binary()' for the detailed mini batch output shape.
    """
    def __init__(self, config:dict):
        self.sr         = config['common']['sr']
        self.data_path  = config['data']['data_path']
        self.max_second = config['common']['train_max_sec']
        self.max_length = int(config['common']['sr'] * config['common']['train_max_sec'])
        self.fix_to_maxlen = config['common']['fix_to_maxlen']        
        self.nb_sample_for_iter = config['common']['nb_steps_eval'] * config['common']['batch_size']

        self.speaker_list, self.sample_dict, _, _ = gather_train_filepaths(config)
                
        # speaker cycle
        random.shuffle(self.speaker_list)
        self.pos_speaker_cycle = cycle(self.speaker_list)

        # paired speakers cycle
        self.neg_speaker_cycle = list(combinations(self.speaker_list, 2))
        random.shuffle(self.neg_speaker_cycle)
        self.neg_speaker_cycle = cycle(self.neg_speaker_cycle)
        
        self.sample_for_iteration()

    def sample_for_iteration(self):
        self.train_samples = []
        for _ in range(self.nb_sample_for_iter):
            label = int(random.random() < .5)
            if label: # label == 1: sample positive pair
                s1 = next(self.pos_speaker_cycle)
                [(_, sample1), (_, sample2)] = random.sample(self.sample_dict[s1], 2)
                
            else:     # label == 0: sample negative pair
                (s1, s2) = next(self.neg_speaker_cycle)
                (_, sample1) = random.choice(self.sample_dict[s1])
                (_, sample2) = random.choice(self.sample_dict[s2])
                
            self.train_samples.append((label, sample1, sample2))
            
    def get_total_dataset_size(self):
        return sum([len(v) for v in self.sample_dict.values()])
                    
    def __len__(self):
        return len(self.train_samples)

    def __getitem__(self, index):
        label, filepath0, filepath1 = self.train_samples[index]
        waveform0, _ = torchaudio.load(opj(self.data_path, filepath0)) # (1, T)
        waveform1, _ = torchaudio.load(opj(self.data_path, filepath1))

        waveform0 = waveform0.squeeze(0) # (T)
        waveform1 = waveform1.squeeze(0)
        
        if len(waveform0) > self.max_length:
            waveform0 = crop_audio(waveform0, opj(self.data_path, filepath0), self.max_second, self.max_length, self.sr)
        elif self.fix_to_maxlen:
            waveform0 = pad_audio(waveform0, self.max_length)            
                            
        if len(waveform1) > self.max_length:
            waveform1 = crop_audio(waveform1, opj(self.data_path, filepath1), self.max_second, self.max_length, self.sr)
        elif self.fix_to_maxlen:
            waveform1 = pad_audio(waveform1, self.max_length)

        return waveform0, waveform1, label


class EvalDataset(Dataset):
    """ Evaluation Dataset for the {Binary Classification | Similarity Comparison} wihtin two audio samples.
    
    Args:
        config (dict) : hyperparameters
        mode   (str)  : 'valid' | 'test'
        
    Attributes:
        data_path (str) : path to the root data folder.
        trials (np.array) : a list of sample pairs  ... [Tuple (label (int), filepath1 (str), filepath2 (str) ), ...]
        
    * see 'pad_collate_binary()' for the detailed mini batch output shape.
    """
    def __init__(self, config:dict, mode:str):        
        self.data_path = config['data']['data_path']
        self.trials = pd.read_csv(config['data'][f'{mode}_trials'])[['LABEL', 'SAMPLE1', 'SAMPLE2']].to_numpy()

    def get_pos_probability(self):
        """ Returns the prior probability of the Positive pair """
        return np.sum(self.trials[:, 0]==1) / len(self.trials)
    
    def __len__(self):
        return len(self.trials)
        
    def __getitem__(self, index):
        label, filepath0, filepath1 = self.trials[index]
        waveform0, _ = torchaudio.load(opj(self.data_path, filepath0))
        waveform1, _ = torchaudio.load(opj(self.data_path, filepath1))

        waveform0 = waveform0.squeeze(0)
        waveform1 = waveform1.squeeze(0)
                
        return waveform0, waveform1, label

