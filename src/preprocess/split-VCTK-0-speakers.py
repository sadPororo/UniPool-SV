import os
import re
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from itertools import product
from os.path import join as opj
from tqdm import tqdm as tqdm


parser = argparse.ArgumentParser(description = "VCTK-Corpus set split")
parser.add_argument('--data_path', type=str, default="./data/VCTK-Corpus/preprocess", help='Source directory')
args = parser.parse_args();


def _generator(iterable):
    for i in iterable: yield i

def remove_whitespace(line:str):
    line = line[line.find(' '):]
    while line[0] == ' ':
        line = line[1:]            
    return line

def binning_columns(column: pd.Series, n_bins: int):
    bin_labels = pd.cut(column, n_bins, retbins=True)[-1]
    bin_labels = ((bin_labels[:-1] + bin_labels[1:]) // 2).astype(int)
    return pd.cut(column, n_bins, labels=bin_labels)
    
if __name__ == "__main__":
    
    # load speaker meta info
    with open(opj(args.data_path, 'speakers', 'speaker-info.txt'), 'r') as f:
        speaker_meta = []
        for line in _generator(f.readlines()[1:]):
            id  = line[:line.find(' ')].strip(); line = remove_whitespace(line)
            age = line[:line.find(' ')].strip(); line = remove_whitespace(line)
            gender  = line[:line.find(' ')].strip(); line = remove_whitespace(line)
            accents = line[:line.find(' ')].strip(); line = remove_whitespace(line)
            region  = re.sub(' +', ' ', line[:line.find('\n')].strip())
            
            speaker_meta.append(['p'+id, age, gender, accents, region])

    speaker_meta = pd.DataFrame(speaker_meta, columns=['ID', 'AGE', 'GENDER', 'ACCENTS', 'REGION']).set_index('ID')
    
    # exclude speaker [p280, p315] of risen technical issues
    speaker_list = speaker_meta.index.to_list()
    if 'p280' in speaker_list:
        speaker_list.remove('p280')
    if 'p315' in speaker_list:
        speaker_list.remove('p315')
    speaker_meta = speaker_meta.loc[speaker_list]
    
    # collect utterance counts for speakers
    sample_counts_per_speaker = {}
    for speaker_id in tqdm(speaker_list):
        sample_counts_per_speaker[speaker_id] = len(os.listdir(opj(args.data_path, 'wav16', speaker_id)))
        
    speaker_meta['COUNTS'] = pd.Series(sample_counts_per_speaker)
    
    # binning too much specified columns [AGE, COUNTS]
    speaker_meta['AGE'] = speaker_meta['AGE'].astype(int)
    speaker_meta['AGE2'] = binning_columns(speaker_meta['AGE'], 2)
    speaker_meta['AGE3'] = binning_columns(speaker_meta['AGE'], 3)
    
    speaker_meta['COUNTS2'] = binning_columns(speaker_meta['COUNTS'], 2)
    speaker_meta['COUNTS3'] = binning_columns(speaker_meta['COUNTS'], 3)

    # Stratified Split (Train | Val-test)
    stratify_conditions = ['AGE3', 'GENDER', 'ACCENTS', 'COUNTS2']

    indices = speaker_meta[stratify_conditions].value_counts().index[
        np.where(speaker_meta[stratify_conditions].value_counts()==1)[0]]
    
    single_instances = np.zeros_like(np.array(speaker_meta.index)).astype(bool)
    for i in range(len(indices)):
        single_instances |= np.array(speaker_meta[stratify_conditions]==indices[i]).all(axis=1)

    df_train, df_val_test = train_test_split(speaker_meta[~single_instances], test_size=0.4, random_state=42, stratify=speaker_meta[~single_instances][stratify_conditions])
    df_train = pd.concat([df_train, speaker_meta[single_instances]])
    
    # Stratified split (valid | test)
    indices = df_val_test[stratify_conditions].value_counts().index[
        np.where(df_val_test[stratify_conditions].value_counts()==1)[0]]
    single_instances = np.zeros_like(np.array(df_val_test.index)).astype(bool)

    for i in range(len(indices)):
        single_instances |= np.array(df_val_test[stratify_conditions]==indices[i]).all(axis=1)

    df_val, df_test = train_test_split(df_val_test[~single_instances], test_size=0.5, random_state=42, stratify=df_val_test[~single_instances][stratify_conditions])
    df_val_test = shuffle(df_val_test[single_instances], random_state=42)
    df_val  = pd.concat([df_val, df_val_test[:len(df_val_test)//2]])
    df_test = pd.concat([df_test, df_val_test[len(df_val_test)//2:]])
    
    # save csv
    save_columns = list(set(['AGE', 'GENDER', 'ACCENTS', 'REGION', 'COUNTS']))
    df_train[save_columns].to_csv(opj(args.data_path, 'speakers', 'speaker-train.csv'))
    df_val[save_columns].to_csv(opj(args.data_path, 'speakers', 'speaker-valid.csv'))
    df_test[save_columns].to_csv(opj(args.data_path, 'speakers', 'speaker-test.csv'))
