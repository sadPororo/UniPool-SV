import os
import random
import argparse

import pandas as pd
import numpy as np
import torchaudio
import torch.nn.functional as F

from itertools import combinations, cycle
from os.path import join as opj
from tqdm import tqdm


parser = argparse.ArgumentParser(description = "VoxCeleb 2: sample balanced trials from evalutation dataset")
parser.add_argument('--data_path', type=str, default="./data/VoxCeleb/preprocess", help='Source directory')
parser.add_argument('--total_sample_size', type=int, default=64000, help='approximate sample size for total number of trial pairs')
args = parser.parse_args();

sr = 16000
mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=23, melkwargs={'win_length':round(sr*0.025), 
                                                                                  'hop_length':round(sr*0.01), 
                                                                                  'f_min':20, 'f_max':3700, 'n_fft':2048, 'n_mels':80})

def _generator(iterable):
    for i in iterable: yield i
    
if __name__ == "__main__":
    
    random.seed(42)
    data_path = opj(args.data_path, 'wav16', 'vox2_test_wav')
    N = args.total_sample_size
    
    speaker_list = list(pd.read_csv(opj(args.data_path, 'speakers', 'speaker-vox2-test.csv'))['ID'])
    
    total_sample_list = []
    pos_speaker_pool = cycle(speaker_list)
    neg_speaker_pool = cycle(list(combinations(speaker_list, 2)))
    
    len_pos_pool = len(speaker_list)
    len_neg_pool = len(list(combinations(speaker_list, 2)))

    N_pos = int(np.ceil(N//2 / len_pos_pool)) * len_pos_pool
    N_neg = int(np.ceil(N//2 / len_neg_pool)) * len_neg_pool

    # mfcc_dict = {}
    # for s_id in _generator(tqdm(speaker_list)):
    #     for v_id in _generator(os.listdir(opj(args.data_path, s_id))):
    #         for f_id in _generator(os.listdir(opj(args.data_path, s_id, v_id))):
    #             if f_id.endswith('.wav'):
    #                 f_path = opj(s_id, v_id, f_id)
    #                 mfcc_dict[f_path] = mfcc_transform(torchaudio.load(opj(args.data_path, f_path))[0]).mean(dim=-1)
    
        
    # Sample positive pairs
    pos_trial_list = []
    for _ in tqdm(range(N_pos)):
        s_id = next(pos_speaker_pool)
        vsrc_match = int(.5 <= random.random())
        
        # Sample from the sample video clip
        if vsrc_match:
            v_id = random.choice(os.listdir(opj(data_path, s_id)))
            
            # At least two audio samples required from the video clip
            while len(os.listdir(opj(data_path, s_id, v_id))) < 2:
                v_id = random.choice(os.listdir(opj(data_path, s_id)))
        
            f_id1, f_id2 = random.sample(os.listdir(opj(data_path, s_id, v_id)), 2)
            
            sample1 = opj(s_id, v_id, f_id1)
            sample2 = opj(s_id, v_id, f_id2)
        
        # Sample from the different video clip
        else:
            v_id1, v_id2 = random.sample(os.listdir(opj(data_path, s_id)), 2)
            f_id1 = random.choice(os.listdir(opj(data_path, s_id, v_id1)))
            f_id2 = random.choice(os.listdir(opj(data_path, s_id, v_id2)))
            
            sample1 = opj(s_id, v_id1, f_id1)
            sample2 = opj(s_id, v_id2, f_id2)
            
        # freq_match = F.cosine_similarity(mfcc_dict[sample1], mfcc_dict[sample2], dim=-1)[0].item()
        
        sample1 = opj('vox2_test_wav', sample1)
        sample2 = opj('vox2_test_wav', sample2)
        # pos_trial_list.append([1, sample1, sample2, vsrc_match, freq_match])
        pos_trial_list.append([1, sample1, sample2, vsrc_match])
        

    # Sample negative pairs
    neg_trial_list = []
    for _ in tqdm(range(N_neg)):
        (s_id1, s_id2) = next(neg_speaker_pool)
        
        v_id1 = random.choice(os.listdir(opj(data_path, s_id1)))
        v_id2 = random.choice(os.listdir(opj(data_path, s_id2)))
        vsrc_match = int(v_id1==v_id2)
        
        f_id1 = random.choice(os.listdir(opj(data_path, s_id1, v_id1)))
        f_id2 = random.choice(os.listdir(opj(data_path, s_id2, v_id2)))
        
        sample1 = opj(s_id1, v_id1, f_id1)
        sample2 = opj(s_id2, v_id2, f_id2)
        
        # freq_match = F.cosine_similarity(mfcc_dict[sample1], mfcc_dict[sample2], dim=-1)[0].item()
        
        sample1 = opj('vox2_test_wav', sample1)
        sample2 = opj('vox2_test_wav', sample2)
        # neg_trial_list.append([0, sample1, sample2, vsrc_match, freq_match])
        neg_trial_list.append([0, sample1, sample2, vsrc_match])
        

    # Save Trials
    # trial_df = pd.DataFrame(pos_trial_list + neg_trial_list, columns=['LABEL', 'SAMPLE1', 'SAMPLE2', 'VSRC_MATCH', 'FREQ_MATCH'])
    trial_df = pd.DataFrame(pos_trial_list + neg_trial_list, columns=['LABEL', 'SAMPLE1', 'SAMPLE2', 'VSRC_MATCH'])
    trial_df.to_csv(opj(args.data_path, 'trials', 'trials-vox2.csv'), index=False)
