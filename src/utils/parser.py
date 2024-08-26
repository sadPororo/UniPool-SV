""" Parse the given command line 

    Use "python main.py -h" to call "arginfo_help()" function and rough preview of how the code works.
"""
import os
import re
import sys
import copy

from ast import literal_eval
from os.path import isdir
from utils.utility import route_dicthierarchy, hypload, setInDict
from utils.loggers import printlog, printarg


def arginfo_help():
    """ Print usage amd argument descriptions, then exit the program
    """
    print('usage: main.py [1-action] [2-data] [3-model] [-h]')
    print('\npositional arguments (required):')
    print('  [1] action:  %s'%("{train,eval,infer,resume}"))
    print('  [2] data  :  %s'%("{VCTK,LibriSpeech,Vox1-Base,Vox2-Base}"))
    print('  [3] model :  %s'%("{ExploreWV2,FinetuneWV2,SincNet,ECAPA-TDNN,X-vector,VQspeaker}"))
    
    print('\noptional arguments in general:')
    print('  -h, --help                 show this help message and exit')
    print('  --quickrun                 quick check for the running experiment on the modification, set as True if given')
    print('  --skiptest                 skip evaluation for the testset during training, set as True if given')    
    print('  --neptune                  log experiment with neptune logger, set as True if given')    
    print('  --workers   WORKERS        the number of cpu-cores to use from dataloader (per each device), defaults to: 4')
    print('  --device   [DEVICE0,]      specify the list of index numbers to use the cuda devices, defaults to: [0]')
    print('  --seed      SEED           integer value of random number initiation, defaults to: 42')
    print('  --eval_path EVAL_PATH      result path to load model on the "action" given as {eval,infer}, defaults to: None')
    print('  --rsum_path RSUM_PATH      result path to resume the training process while the "action" given as "resume", defaults to: None')
    print('  --description DESCRIPTION  user parameter for specifying certain version, Defaults to "Untitled".')    
    
    print('\nkeyword arguments:')
    print('  --kwarg KWARG            dynamically modifies any of the hyperparameters declared in ../configs/.../...*.yaml or ./benchmarks/...*.yaml')
    print('  (e.g.) --lr 0.001 --batch_size 64 --nb_total_step 25000 ...')
    
    sys.exit()
    

def argupdate(config:dict, NUM_POS_ARG:int):
    """ Parse the given **argv and update optional/keyword arguments 

    Args:
        config (dict): a python dictionary with a multi-level hierarchy
        NUM_POS_ARG (int): number of positional arguments, used to slice the command line
        
    Returns:
        config (dict): an updated dictionary
    """
    
    kwargs = re.split(' *--', ' '.join(sys.argv[NUM_POS_ARG+1:]))[1:]
    paths, types, _ = route_dicthierarchy(config)

    for term in copy.deepcopy(kwargs):
        arg = re.split(' ', term, maxsplit=1)
        assert (len(arg)==1) or (len(arg)==2), f"\n\tCannot parse the given term, {term}"
        kw = arg[0].strip()
        
        for i, p in enumerate(paths):
            # keyword argument cannot change the positional arguments
            if p[0] == 'posarg': continue
            
            # find the keyword match
            if p[-1] == kw:
                
                # update store_true arguments
                if types[i] == bool:
                    assert len(arg)==1, f"\n\tValue passed to a store_true option, --{arg[0]} {arg[1:]}"
                    setInDict(config, p, True)
                
                else:
                    assert len(arg)==2, f"\n\tArgument '--{arg[0]}' expects value to be passed"
                    val = arg[1].strip()
                    
                    # value type check
                    if bool(re.match(r'^[-+]?[0-9]+$', val)): val = int(val)
                    elif bool(re.match(r'[-+]?[0-9]+\.[0-9]*', val)): val = float(val)
                    elif bool(re.match(r'[-+]?[0-9]+e[-+]?[0-9]*', val)): val = float(val)
                    elif bool(re.match(r'\[.*\]', val)): val = literal_eval(val)
                    elif types[i] == str: pass 
                    
                    # type compatibility check
                    assert types[i] == type(val), f"\n\tType mismatch: '--{arg[0]}' requires '{types[i].__name__}' type argument, but '{type(val).__name__}' has given: {arg[1:]}"
                    setInDict(config, p, val)

                # pop the term from the argv if the update proceeded
                while term in kwargs: kwargs.remove(term)
    
    assert len(kwargs)==0, f"\n\tUnmatched keyword arguments exist: {['--'+term for term in kwargs]}"
    
    # Override device configuration if CUDA_VISIBLE_DEVICES has given
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        config['general']['device'] = list(range(len(os.environ.get('CUDA_VISIBLE_DEVICES').split(','))))

    return config


def get_config():
    """ Load hyperparameters and update the arguments if configured in command line 

    Returns:
        config (dict): a python dictionary of hyperparameters for running experiment
        FLAG_LOAD_CONFIG (bool): returns whether subjective model path has given. Will be set True when resuming the training or model evaluation
    """
    # print argument info if 'help' keyword is given
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        arginfo_help()

    config = {}
    
    # parse the positional arguments first
    config['posarg'] = {}
    assert sys.argv[1] in ['train', 'eval', 'infer', 'resume'], "\n\tGiven the 1st argument (action) '%s' is not supported, try among {train,eval,infer,resume}"%(sys.argv[1])
    config['posarg']['action'] = sys.argv[1] 
    if sys.argv[1] == 'train':
        assert sys.argv[2] in ['VCTK', 'LibriSpeech', 'Vox1-Base', 'Vox2-Base', '_'], "\n\tGiven the 2nd argument (data) '%s' is not supported, try among {VCTK,LibriSpeech,Vox1-Base,Vox2-Base}"%(sys.argv[2])
        # assert sys.argv[3] in ['ExploreWV2', 'FinetuneWV2', 'SincNet', 'ECAPA-TDNN', 'X-vector', '_'], "\n\tGiven the 3rd argument (model) '%s' is not supported, try among {ExploreWV2,FinetuneWV2,SincNet}"%(sys.argv[3])
    else: pass
    config['posarg']['data']   = sys.argv[2]
    config['posarg']['model']  = sys.argv[3]
    
    NUM_POS_ARG = len(config['posarg'])
    
    # set defaults to the optional arguments
    config['general'] = {}
    config['general']['quickrun']  = False
    config['general']['skiptest']  = False
    config['general']['skipDCF']   = False
    config['general']['neptune']   = False
    config['general']['workers']   = 4
    config['general']['device']    = [0]
    config['general']['seed']      = 42
    config['general']['eval_path'] = ''
    config['general']['eval_data'] = ''
    config['general']['rsum_path'] = ''
    config['general']['description'] = 'Untitled'

    # load data/model/train hyperparameters corresponding to the positional arguments
    if config['posarg']['data'] != '_':
        config['data']  = hypload('../configs/data/data-%s-config.yaml'%(config['posarg']['data']))
    if config['posarg']['model'] != '_':
        config['model'] = hypload('./benchmarks/%s/model-config.yaml'%(config['posarg']['model']))       
    config['common'] = hypload('../configs/common/training-base-config.yaml')

    # update the configurations if the argument is passed
    config = argupdate(config, NUM_POS_ARG)
    
    # load neptune configuration if '--neptune' is True
    if config['general']['neptune']:
        config['neptune'] = hypload('../configs/neptune/neptune-logger-config.yaml')
        
    # check constraints depending on the 'action'
    if config['posarg']['action'] == 'train':
        assert (config['general']['eval_path']==''), f"\n\tGiven positional argument [1] (action) 'train' does not support using argument '--eval_path'"
        assert (config['general']['rsum_path']==''), f"\n\tGiven positional argument [1] (action) 'train' does not support using argument '--rsum_path'"

    elif config['posarg']['action'] == 'eval':
        assert (config['general']['eval_path']!=''), f"\n\tGiven positional argument [1] (action) 'eval' requires '--eval_path' to be set"
        assert (config['general']['rsum_path']==''), f"\n\tGiven positional argument [1] (action) 'eval' does not support using argument '--rsum_path'"
        assert (isdir(f"../res/{config['general']['eval_path']}")), f"\n\tGiven '--eval_path' {config['general']['eval_path']} is not a directory"
        
    elif config['posarg']['action'] == 'infer':
        raise NotImplementedError('infer')
        
    elif config['posarg']['action'] == 'resume':
        assert (config['general']['rsum_path']!=''), f"\n\tGiven positional argument [1] (action) 'resume' requires '--rsum_path' to be set"
        assert (config['general']['eval_path']==''), f"\n\tGiven positional argument [1] (action) 'resume' does not support using argument '--eval_path'"
        assert (isdir(config['general']['rsum_path'])), f"\n\tGiven '--rsum_path' {config['general']['rsum_path']} is not a directory"
    
    return config


def load_eval_config(config):
    """ """
    # load the configuration
    load_path = config['general']['eval_path']
    assert isdir(f'../res/{load_path}'), f"\n\tGiven '--eval_path' {load_path} is not a directory"
    load_conf = hypload(f'../res/{load_path}/config.yaml')
    
    # update positional argument (action)
    load_conf['posarg']['action'] = config['posarg']['action']
    
    # update general argument (device)
    load_conf['general']['device'] = config['general']['device']
    
    # update the evaluation data path
    if config['general']['eval_data'] != '':
        load_conf['general']['eval_data'] = config['general']['eval_data']
        
        data_conf = hypload('../configs/data/data-%s-config.yaml'%(config['general']['eval_data']))
        load_conf['data']['data_path']     = data_conf['data_path']
        load_conf['data']['valid_trials']  = data_conf['valid_trials']
        load_conf['data']['test_trials']   = data_conf['test_trials']
        load_conf['data']['nb_class_val']  = data_conf['nb_class_val']
        load_conf['data']['nb_class_test'] = data_conf['nb_class_test']
    
    else:
        load_conf['general']['eval_data'] = load_conf['posarg']['data']
            
    return load_conf
    
        
    
        
        
    
    
    