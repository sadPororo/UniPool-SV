import os
import argparse

import pandas as pd

import torch.nn.functional as F
import torchaudio

from itertools import combinations
from os.path import join as opj
from tqdm import tqdm as tqdm


parser = argparse.ArgumentParser(description = "VCTK-Corpus compute all trials from evalutation dataset")
parser.add_argument('--data_path', type=str, default="./data/VCTK-Corpus/preprocess", help='Source directory')
args = parser.parse_args();


def _generator(iterable):
    for i in iterable: yield i

# sample_rate = 16000
# mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=23, melkwargs={'win_length':round(sample_rate*0.025), 
#                                                                                            'hop_length':round(sample_rate*0.01), 
#                                                                                            'f_min':20, 'f_max':3700, 'n_fft':2048, 'n_mels':80})
    
if __name__ == "__main__":
    
    # get all combinations from valid / test speaker sets each.
    for split in ['valid', 'test']:
        df = pd.read_csv(opj(args.data_path, 'speakers', 'speaker-%s.csv'%split), index_col='ID').to_dict('index')
        
        sample_list = []
        mfcc_dict   = {}
        for speaker_id in _generator(list(df.keys())):
            for wav_id in _generator(os.listdir(opj(args.data_path, 'wav16', speaker_id))):
                if wav_id.endswith('.wav'):
                    sample_list.append(opj(speaker_id, wav_id))
                    # mfcc_dict[opj(speaker_id, wav_id)] = mfcc_transform(torchaudio.load(opj(args.data_path, 'wav16', speaker_id, wav_id))[0]).mean(dim=-1)
                    
        trial_list = []
        
        for (sample1, sample2) in _generator(tqdm(list(combinations(sample_list, 2)))):
            id1 = sample1.split('/')[0]
            id2 = sample2.split('/')[0]

            # label
            label = 1 if id1 == id2 else 0
            
            # sample_match
            sample_match = 1 if sample1 == sample2 else 0
            
            # gender_match
            gender_match = 1 if df[id1]['GENDER'] == df[id2]['GENDER'] else 0
                
            # hard_age_match
            hard_age_match = 1 if df[id1]['AGE'] == df[id2]['AGE'] else 0
                
            # soft_age_match
            soft_age_match = 1 if df[id1]['AGE3'] == df[id2]['AGE3'] else 0
                
            # accent_match
            accent_match = 1 if df[id1]['ACCENTS'] == df[id2]['ACCENTS'] else 0
            
            # region_match
            region_match = 1 if df[id1]['REGION'] == df[id2]['REGION'] else 0
            
            # # freq_match
            # freq_match = F.cosine_similarity(mfcc_dict[sample1], mfcc_dict[sample2], dim=-1)[0].item()

            
            # trial_list.append([label, sample1, sample2, sample_match, 
            #                     gender_match, hard_age_match, soft_age_match, accent_match, region_match, freq_match])
            trial_list.append([label, sample1, sample2, sample_match, 
                                gender_match, hard_age_match, soft_age_match, accent_match, region_match])
                
        # trial_list = pd.DataFrame(trial_list, columns=['LABEL', 'SAMPLE1', 'SAMPLE2', 'SAMPLE_MATCH', 
        #                                                'GENDER_MATCH', 'HARD_AGE_MATCH', 'SOFT_AGE_MATCH', 'ACCENT_MATCH', 'REGION_MATCH', 'FREQ_MATCH'])
        trial_list = pd.DataFrame(trial_list, columns=['LABEL', 'SAMPLE1', 'SAMPLE2', 'SAMPLE_MATCH', 
                                                       'GENDER_MATCH', 'HARD_AGE_MATCH', 'SOFT_AGE_MATCH', 'ACCENT_MATCH', 'REGION_MATCH'])
        
        trial_list.to_csv(opj(args.data_path, 'trials', 'trials-%s-raw.csv'%split), index=False)

