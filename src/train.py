""" This is a wrapper module to initiate the training of benchmarks

[Summary]
- DDP/Single GPU setups
- initiate the offline/online logger objects
- import the benchmark package in configuration and run the trainer
"""
import os
import sys
import importlib

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from os.path import join as opj
from utils.utility import random_state_init, get_random_state, hypwrite
from utils.loggers import init_loggings, printlog


def ddp_setup(is_master):
    """ Disables printing when not the process is not the master """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def ddp_init(rank:int, config:dict):
    """ Single node / multi-gpu processing initiation

    Args:
        rank (int): local rank of GPU in node
        config (dict): hyperparameters
        
    Returns:
        local_gpu_id (int): gpu-id for the current rank process
        config (dict): hyperparameters, see "init_loggings()" description for the configuration updates
        logger: (neptune.run | None): online logger object if the process rank is 0 (master process)
    """
    # Set GPU device for the current subprocess
    local_gpu_id = int(config['general']['device'][rank])
    torch.cuda.set_device(local_gpu_id)

    # Initiate loggers
    if config['posarg']['action'] == 'train':
        config, logger = init_loggings(config, rank==0)
    else:
        logger = None
    logging_path = config['general']['logging_path']

    # Initiate distributed
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23456',
                            world_size=config['general']['world_size'],
                            rank=rank)
    random_state_init(seed=config['general']['seed'], device_list=[local_gpu_id])
    
    # print device of the process
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        printlog(f"...GPU device (local-id: {local_gpu_id}, global-id: {os.environ.get('CUDA_VISIBLE_DEVICES').split(',')[rank]}): ready on rank {rank}.", logging_path)    
    else:
        printlog(f'...GPU device {local_gpu_id}: ready on rank {rank}.', logging_path)    
    dist.barrier()
    
    # disable print(sys.stdout) if not the master (rank==0)
    ddp_setup(rank==0); printlog('', config['general']['logging_path'], rank==0)
    
    return local_gpu_id, config, logger
    

def ddp_train(rank:int, config:dict):
    """ main function for training benchmark via DDP.

    Args:
        rank (int): rank of the process
        config (dict): hyperparameters
    """
    local_gpu_id, config, logger = ddp_init(rank, config)
    logging_path = config['general']['logging_path']
    is_master    = (rank==0)
    
    printlog('Initiating training process...', logging_path, is_master)
    printlog("...using package: '.benchmarks.%s'"%config['posarg']['model'], logging_path, is_master)    
    Trainer = importlib.import_module('trainer', package='benchmarks.%s'%config['posarg']['model'])
    getattr(Trainer, 'train')(config, logger, device_id=local_gpu_id)
    
    dist.destroy_process_group()
    
    return 0


def train_main(config:dict):
    """ main function to initiate training process

    Args:
        config (dict): hyperparameters
    """
   
    # Add package path
    sys.path.append('./src/benchmarks/%s'%config['posarg']['model'])

    # Set every random states on devices
    random_state_init(seed=config['general']['seed'], device_list=config['general']['device'])
        
    # multi GPU processing setup
    if len(config['general']['device']) > 1:
        config['general']['world_size'] = len(config['general']['device'])
        mp.spawn(ddp_train,
                 args=(config,),
                 nprocs=config['general']['world_size'],
                 join=True)
        
    # single GPU setup
    elif len(config['general']['device']) == 1:
        config['general']['world_size'] = 1
        config, logger = init_loggings(config)

        logging_path = config['general']['logging_path']
        local_gpu_id = int(config['general']['device'][0])
        if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
            printlog(f"...GPU device (local-id: {local_gpu_id}, global-id: {os.environ.get('CUDA_VISIBLE_DEVICES')}): ready.\n", logging_path)    
        else:
            printlog(f'...GPU device {local_gpu_id}: ready.\n', logging_path)
        printlog('Initiating training process...', logging_path)
        printlog("...using package: '.benchmarks.%s'"%config['posarg']['model'], logging_path)    
        Trainer = importlib.import_module('trainer', package='benchmarks.%s'%config['posarg']['model'])
        getattr(Trainer, 'train')(config, logger, device_id=local_gpu_id)
            
    # else use CPU
    else:
        raise NotImplementedError(config['general']['device'])

    
# if __name__ == "__main__":
#     config= {

#         'posarg': {
#             'action'           : 'train',
#             'data'             : 'VCTK',
#             'model'            : 'ExploreWV2'
#             },

#         'general': {
#             'quickrun'         : False,
#             'neptune'          : False,
#             'workers'          : 4,
#             'device'           : [0],
#             'seed'             : 1029,
#             'eval_path'        : '',
#             'rsum_path'        : ''
#             },

#         'data': {
#             'data_path'      : '/home/jinsob/AudioDatasets/VCTK/data',
#             'valid_trials'   : '/home/jinsob/AudioDatasets/VCTK/meta/trials-valid-B.csv',
#             'test_trials'    : '/home/jinsob/AudioDatasets/VCTK/meta/trials-test-B.csv',
#             'speaker_meta'   : '/home/jinsob/AudioDatasets/VCTK/meta',
#             'nb_class_train' : 72,
#             'nb_class_test'  : 18,
#             'nb_class_val'   : 18,
#             },

#         'model': {
#             'w2v_model'        : 'base',
#             'use_pretrain'     : True,
#             'bsz_ratio'        : 2,
#             'optim'            : 'Adam',
#             'scheduler'        : 'OneCycleLR',
#             'max_lr'           : 0.005,
#             'anneal_strategy'  : 'linear',
#             'total_steps'      : 13000,
#             'warmup_steps'     : 6000,
#             'w2v_freeze_steps' : 10000
#             },

#         'common': {
#             'nb_epochs'        : 30,
#             'batch_size'       : 8
#             } 
#         }
    
#     train_main(config)