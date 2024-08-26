import os
import argparse
import itertools
import torchaudio

from tqdm import tqdm as tqdm
from multiprocessing import Pool, freeze_support

from os.path import join as opj


resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)

parser = argparse.ArgumentParser(description = "VCTK-Corpus preprocessing")
parser.add_argument('--read_path', type=str, default="./data/VCTK-Corpus", help='Source directory')
parser.add_argument('--save_path', type=str, default=None, help='Output directory, default to "{READ_PATH}/preprocess/"')
parser.add_argument('--n_workers', type=int, default=4, help='Multi-processing Cores')
args = parser.parse_args();


def _generator(iterable):
    for i in iterable: yield i
    

def preprocess_resample(filepath_list):
    filepath, read_data_path, save_data_path = filepath_list
    
    # load audio file
    waveform, sr = torchaudio.load('/'.join([read_data_path, filepath]))

    # VCTK: 48K sample_rate >> 16K sample_rate
    if sr != 16000:
        waveform = resampler(waveform)
        sr = 16000
        
    # save 16K audio
    torchaudio.save(filepath='/'.join([save_data_path, filepath]), src=waveform, sample_rate=sr)    


if __name__ == "__main__":
    freeze_support()

    # Define output directory
    if args.save_path is None:
        save_path = opj(args.read_path, 'preprocess')
    else:
        save_path = args.save_path
    
    speaker_list = os.listdir(opj(args.read_path, 'wav48'))
    # remove speakers (p280, p315)
    if 'p280' in speaker_list:
        speaker_list.remove('p280')
    if 'p315' in speaker_list:
        speaker_list.remove('p315')
        
    # gather total file list to preprocess
    total_filepath_list = []
    for speaker_id in _generator(tqdm(speaker_list)):
        os.makedirs(opj(save_path, 'wav16', speaker_id), exist_ok=True)
        
        for wav_id in _generator(os.listdir(opj(args.read_path, 'wav48', speaker_id))):
            if wav_id.endswith('.wav'):
                
                if int(wav_id.split('_')[1][:3]) > 24:
                    total_filepath_list.append(opj(speaker_id, wav_id))
                    
    # resample to 16K
    with Pool(args.n_workers) as pool:
        list(tqdm(pool.imap(preprocess_resample, zip(total_filepath_list, 
                                                 itertools.repeat(opj(args.read_path, 'wav48')), 
                                                 itertools.repeat(opj(save_path, 'wav16')))), total=len(total_filepath_list)))
            
        
        