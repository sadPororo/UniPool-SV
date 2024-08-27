import os
import argparse

import pandas as pd

from itertools import combinations
from os.path import join as opj
from tqdm import tqdm as tqdm

parser = argparse.ArgumentParser(description = "VoxCeleb set split")
parser.add_argument('--data_path', type=str, default="./data/VoxCeleb/preprocess", help='Source directory')
args = parser.parse_args();



def _generator(iterable):
    for i in iterable: yield i

def get_speaker_subset(spkr_id):
    for k, v in speaker_dict.items():
        if spkr_id in v:
            return k
    
if __name__ == "__main__":
    
    # speaker list
    speaker_dict = {}
    for subset in ['vox1_dev_wav', 'vox1_test_wav', 'vox2_dev_wav', 'vox2_test_wav']:
        speaker_dict[subset] = []

        for speaker_id in tqdm(os.listdir(opj(args.data_path, 'wav16', subset))):
            speaker_dict[subset].append(speaker_id)
            
        pd.DataFrame(speaker_dict[subset], columns=['ID']).to_csv(opj(args.data_path, 'speakers', 'speaker-' + subset[:-4].replace('_', '-') + '.csv'), index=False)

    # print(speaker_dict.keys())
    # Vox1-O: (.txt) to (.csv)
    with open(opj(args.data_path, 'trials', 'vox1_sv_trial_pairs.txt')) as f:
        trial_list = [line.strip().split(' ') for line in f.readlines()]
        
    vox1_trial_df = pd.DataFrame(trial_list, columns=['LABEL', 'SAMPLE1', 'SAMPLE2'])
    print(vox1_trial_df)
    vox1_trial_df['LABEL']   = vox1_trial_df['LABEL'].astype(int)
    vox1_trial_df['SAMPLE1'] = vox1_trial_df['SAMPLE1'].apply(lambda x: opj(get_speaker_subset(x.split('/')[0]), x))
    vox1_trial_df['SAMPLE2'] = vox1_trial_df['SAMPLE2'].apply(lambda x: opj(get_speaker_subset(x.split('/')[0]), x))

    vox1_trial_df.to_csv(opj(args.data_path, 'trials', 'trials-vox1.csv'), index=False)
