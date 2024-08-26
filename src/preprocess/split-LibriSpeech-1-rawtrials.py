import os
import re
import argparse

import pandas as pd
import torch.nn.functional as F
import torchaudio

from itertools import combinations
from os.path import join as opj
from tqdm import tqdm as tqdm

parser = argparse.ArgumentParser(description = "LibriSpeech set split")
parser.add_argument('--data_path', type=str, default="./data/LibriSpeech/preprocess", help='Source directory')
args = parser.parse_args();


# sample_rate = 16000
# mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=23, melkwargs={'win_length':round(sample_rate*0.025), 
#                                                                                            'hop_length':round(sample_rate*0.01), 
#                                                                                            'f_min':20, 'f_max':3700, 'n_fft':2048, 'n_mels':80})

def _generator(iterable):
    for i in iterable: yield i

    
if __name__ == "__main__":
    
    # load speaker meta info
    with open(opj(args.data_path, 'speakers', 'SPEAKERS.TXT'), 'r') as f:
        speaker_meta = []
        for line in f:    
            if line.startswith(';'):
                pass
            else:
                # speaker_meta.append(line.strip())
                speaker_meta.append(re.sub(' +\| +', '\t', line.strip()).split('\t'))

    df = pd.DataFrame(speaker_meta, columns=['ID', 'SEX', 'SUBSET', 'MINUTES', 'NAME']).set_index('ID')
    df.to_csv(opj(args.data_path, 'meta', 'speaker-meta.csv'))
    df = df.to_dict('index')

    # join clean & others path list
    for split1 in ['dev', 'test']:
        
        sample_list = []
        # mfcc_dict   = {}
        
        for split2 in ['clean', 'other']:
            split_dir = '-'.join([split1, split2]) # dev-clean | dev-other | test-clean | test-other
            
            for speaker_id in _generator(os.listdir(opj(args.data_path, 'wav16', split_dir))):
                for chapter_id in _generator(os.listdir(opj(args.data_path, 'wav16', split_dir, speaker_id))):
                    for wav_id in _generator(os.listdir(opj(args.data_path, 'wav16', split_dir, speaker_id, chapter_id))):
                        if wav_id.endswith('.wav'):
                            filepath = opj(split_dir, speaker_id, chapter_id, wav_id)
                            sample_list.append(filepath)
                            # mfcc_dict[filepath] = mfcc_transform(torchaudio.load(opj(args.data_path, 'wav16', filepath))[0]).mean(dim=-1)

        trial_list = []

        for (sample1, sample2) in _generator(tqdm(list(combinations(sample_list, 2)))):
            id1 = sample1.split('/')[1]
            id2 = sample2.split('/')[1]

            # label
            label = 1 if id1 == id2 else 0
            
            # sample_match
            sample_match = 1 if sample1 == sample2 else 0
            
            # gender_match
            gender_match = 1 if df[id1]['SEX'] == df[id2]['SEX'] else 0
                
            # freq_match
            # freq_match = F.cosine_similarity(mfcc_dict[sample1], mfcc_dict[sample2], dim=-1)[0].item()

            
            # trial_list.append([label, sample1, sample2, sample_match, gender_match, freq_match])
            trial_list.append([label, sample1, sample2, sample_match, gender_match])
                
        # trial_list = pd.DataFrame(trial_list, columns=['LABEL', 'SAMPLE1', 'SAMPLE2', 'SAMPLE_MATCH', 'GENDER_MATCH', 'FREQ_MATCH'])
        trial_list = pd.DataFrame(trial_list, columns=['LABEL', 'SAMPLE1', 'SAMPLE2', 'SAMPLE_MATCH', 'GENDER_MATCH'])
        
        trial_list.to_csv(opj(args.data_path, 'trials', 'trials-%s-raw.csv'%split1), index=False)
