import os
import argparse
import itertools
import subprocess

from tqdm import tqdm as tqdm
from multiprocessing import Pool, freeze_support

from os.path import join as opj


parser = argparse.ArgumentParser(description = "LibriSpeech preprocessing")
parser.add_argument('--read_path', type=str, default="./data/LibriSpeech", help='Source directory')
parser.add_argument('--save_path', type=str, default=None, help='Output directory, default to "{READ_PATH}/preprocess/"')
parser.add_argument('--n_workers', type=int, default=4, help='Multi-processing Cores')
args = parser.parse_args();


def _generator(iterable):
    for i in iterable: yield i
    
def preprocess_conversion(filepath_list):
    local_path, read_path, save_path = filepath_list
    
    out = subprocess.call('ffmpeg -i %s %s -c:a pcm_s16le -y >/dev/null 2>&1'%(opj(read_path, local_path), opj(save_path, local_path.replace('.flac', '.wav'))), shell=True)
    # out = subprocess.call('ffmpeg -i %s %s -y >/dev/null 2>&1'%(opj(read_path, local_path), opj(save_path, local_path.replace('.flac', '.wav'))), shell=True)
    if out != 0:
        raise ValueError('Conversion failed %s.'% opj(read_path, local_path))
    

if __name__ == "__main__":
    freeze_support()

    # Define output directory
    if args.save_path is None:
        save_path = opj(args.read_path, 'preprocess')
    else:
        save_path = args.save_path
        
    subset_list = ['dev-clean', 'dev-other', 'test-clean', 'test-other', 'train-clean-100', 'train-clean-360', 'train-other-500']
    for subset in subset_list:
        print('Processing on Subset: %s' % subset)
        
        # gather total file list to preprocess
        print('Gathering Source (Speaker/Chapter/Audio) paths...')
        total_filepath_list = []
        for speaker_id in _generator(tqdm(os.listdir(opj(args.read_path, subset)))):
            
            for chapter_id in _generator(os.listdir(opj(args.read_path, subset, speaker_id))):
                os.makedirs(opj(save_path, subset, speaker_id, chapter_id), exist_ok=True)
                
                for wav_id in _generator(os.listdir(opj(args.read_path, subset, speaker_id, chapter_id))):
                    if wav_id.endswith('.flac'):
                        total_filepath_list.append(opj(subset, speaker_id, chapter_id, wav_id))
                        
        # resample to 16K
        print('Converting (.flac) to (.wav) files...')    
        with Pool(args.n_workers) as pool:
            list(tqdm(pool.imap(preprocess_conversion, zip(total_filepath_list, 
                                                           itertools.repeat(opj(args.read_path)), 
                                                           itertools.repeat(opj(save_path)))), total=len(total_filepath_list)))
