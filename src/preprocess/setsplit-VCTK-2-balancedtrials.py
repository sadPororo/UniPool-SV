import copy
import random
import argparse

import pandas as pd
import numpy as np

from itertools import product
from os.path import join as opj
from tqdm import tqdm as tqdm


parser = argparse.ArgumentParser(description = "VCTK-Corpus compute all trials from evalutation dataset")
parser.add_argument('--data_path', type=str, default="./data/VCTK-Corpus/preprocess", help='Source directory')
parser.add_argument('--min_sample_size', type=int, default=8000, help='approximate sample size for each positive/negative pairs')
args = parser.parse_args();


def _generator(iterable):
    for i in iterable: yield i
    
if __name__ == "__main__":
    
    random.seed(42)
    
    for split in ['valid', 'test']:
        print(f'Running on {split} split...')
        
        # df_eval_raw = pd.read_csv(opj(args.data_path, 'trials', f'trials-{split}-raw.csv'))
        # df_eval_raw['SPEAKER1'] = df_eval_raw['SAMPLE1'].apply(lambda x: x.split('/')[0])
        # df_eval_raw['SPEAKER2'] = df_eval_raw['SAMPLE2'].apply(lambda x: x.split('/')[0])
        # df_eval_raw['SAMPLE1_NUM'] = df_eval_raw['SAMPLE1'].apply(lambda x: int(x[:-4].split('_')[-1]))
        # df_eval_raw['SAMPLE2_NUM'] = df_eval_raw['SAMPLE2'].apply(lambda x: int(x[:-4].split('_')[-1]))
        # df_eval = df_eval_raw.loc[(df_eval_raw['SAMPLE1_NUM'] > 24) & (df_eval_raw['SAMPLE2_NUM'] > 24)]

        df_eval = pd.read_csv(opj(args.data_path, 'trials', f'trials-{split}-raw.csv'))
        df_eval_meta = pd.read_csv((opj(args.data_path, 'speakers', f'speaker-{split}.csv')))
        

        # Get ratios of matching features first
        print('Checking Label/Condition Distributions...\n\t(LABEL, GENDER_MATCH, ACCENT_MATCH)')
        case_ratio_entry = {}
        for cond in product([0], [1, 0], [1, 0]):
            label_idx  = ((df_eval['LABEL']==1)==cond[0])
            gender_idx = ((df_eval['GENDER_MATCH']==1)==cond[1])
            accent_idx = ((df_eval['ACCENT_MATCH']==1)==cond[2])
        
            df_sample = df_eval.loc[label_idx & gender_idx & accent_idx]
            print('\tNEG_SAMPLE:', cond, round(len(df_sample)/len(df_eval)*100, 2), '%')
            case_ratio_entry[''.join([str(i) for i in cond])] = len(df_sample)/len(df_eval)
            
        cond = [1, 1, 1]
        label_idx  = ((df_eval['LABEL']==1)==cond[0])
        gender_idx = ((df_eval['GENDER_MATCH']==1)==cond[1])
        accent_idx = ((df_eval['ACCENT_MATCH']==1)==cond[2])

        df_sample = df_eval.loc[label_idx & gender_idx & accent_idx]
        print('\tPOS_SAMPLE:', cond, round(len(df_sample)/len(df_eval)*100, 2), '%')
        case_ratio_entry[''.join([str(i) for i in cond])] = len(df_sample)/len(df_eval)

        # Pairs for negative - minimum 8000 sample
        NEG_SAMPLE_INDICE = []

        # Sample negatives
        df_negative = df_eval.loc[(df_eval['LABEL']==0)]

        condition_pool = list(product([1, 0], [1, 0]))
        cond_sample_size = np.ceil(args.min_sample_size / len(condition_pool)).astype(int) # ~ 2000
        print('- Sampling negatives')
        
        for cond in condition_pool:
            gender_idx = ((df_negative['GENDER_MATCH']==1)==cond[0])
            accent_idx = ((df_negative['ACCENT_MATCH']==1)==cond[1])
            condition_sample = df_negative.loc[gender_idx & accent_idx]
            
            speaker_pool = list(condition_sample.groupby(['SPEAKER1', 'SPEAKER2']).size().index)
            spk_sample_size = np.ceil(cond_sample_size / len(speaker_pool)).astype(int)
            print(f'Sampling with the condition: \n\tGENDER MATCH - {bool(cond[0])}, ACCENT_MATCH - {bool(cond[1])}')
            print(f'{len(speaker_pool)} speaker combinations exists, {spk_sample_size} samples will be chosen from each... (total {len(speaker_pool)*spk_sample_size} samples)')
            
            for s1_id, s2_id in _generator(tqdm(speaker_pool)):
                speaker_idx = (condition_sample[['SPEAKER1', 'SPEAKER2']]==(s1_id, s2_id)).all(axis=1) | (condition_sample[['SPEAKER1', 'SPEAKER2']]==(s2_id, s1_id)).all(axis=1)
                speaker_sample = condition_sample.loc[speaker_idx]

                # Sample negative pairs from here
                sampled_indice = random.sample(list(speaker_sample.index), spk_sample_size)
                NEG_SAMPLE_INDICE += sampled_indice

        print(f'...Total {len(NEG_SAMPLE_INDICE)} negative trials are sampled!')
        
        # Sample positives
        print('- Sampling positives')
        
        POS_SAMPLE_INDICE = []
        df_positive = df_eval.loc[(df_eval['LABEL']==1)]
        speaker_pool = list(df_positive['SPEAKER1'].unique())
        spk_sample_size = np.ceil(args.min_sample_size / len(speaker_pool)).astype(int)
        print(f'{len(speaker_pool)} unique speaker exists, {spk_sample_size} samples will be chosen from each...')
        
        for s_id in _generator(tqdm(speaker_pool)):
            speaker_sample = df_positive.loc[df_positive['SPEAKER1']==s_id]

            # Sample positive pairs from here
            sampled_indice = random.sample(list(speaker_sample.index), spk_sample_size)
            POS_SAMPLE_INDICE += sampled_indice

        print(f'...Total {len(POS_SAMPLE_INDICE)} negative trials are sampled!')

        df_eval_balanced = copy.deepcopy(df_eval.iloc[sorted(NEG_SAMPLE_INDICE + POS_SAMPLE_INDICE)])
        df_eval_balanced.drop(columns=['SPEAKER1', 'SPEAKER2', 'SAMPLE1_NUM', 'SAMPLE2_NUM'], inplace=True)
        df_eval_balanced.to_csv(opj(args.data_path, f'trials-{split}.csv'), index=False)
        
        print(f'{split} split: total {len(df_eval_balanced)} trials are sampled!')
        print('...Done!\n')
