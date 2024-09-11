""" Utility functions for others """
import os
import torch
import random
import operator

import yaml
import ruamel.yaml
import numpy as np

from functools import reduce
from collections import Iterable



def random_state_init(seed:int=42, device_list:list=[0]):    
    """ Initiate the randomness of machines

    Args:
        seed (int, optional): An integer to start the randomness. Defaults to 42.
        device_list (list, optional): a list of gpu-device ids to use. Defaults to [0]. 
                                    (e.g.) [] -> CPU processing
                                           [0] -> single GPU processing
                                           [0,1,2,3] -> multi GPU processing
    """
    # CPU environ
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # GPU environ
    for gpu_id in device_list:
        torch.cuda.set_device(gpu_id)
        torch.cuda.manual_seed(seed)
    torch.cuda.set_device(device_list[0])
    if len(device_list) == torch.cuda.device_count():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def random_state_recover(device_list:list, **rng_state):
    """ Recover the random state of configured machines

    Args:
        device_list (list): a list of gpu-device ids to use.
                        (e.g.) [] -> CPU processing
                               [0] -> single GPU processing
                               [0,1,2,3] -> multi GPU processing
    """
    # CPU environ
    random.setstate(rng_state['rand_state'])
    np.random.set_state(rng_state['np_state'])
    torch.random.set_rng_state(rng_state['torch_state'])
    os.environ['PYTHONHASHSEED'] = str(rng_state['os_hash_state'])
    
    # GPU environ
    for rank, gpu_id in enumerate(device_list):
        torch.cuda.set_rng_state(rng_state['cuda_state'][rank], device=gpu_id)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_random_state(device_list:list):
    """ Get the current random state from machines

    Args:
        device_list (list): a list of gpu-device ids which are currently online. 

    Returns:
        _type_: _description_
    """
    # CPU environ
    rand_state = random.getstate()
    np_state   = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    os_hash_state = str(os.environ['PYTHONHASHSEED'])
    
    # GPU environ
    cuda_state = {}
    for rank, gpu_id in enumerate(device_list):
        cuda_state[rank] = torch.cuda.get_rng_state(device=gpu_id)
            
    return {'rand_state'   : rand_state, 
            'np_state'     : np_state, 
            'torch_state'  : torch_state, 
            'os_hash_state': os_hash_state,
            'cuda_state'   : cuda_state}


def route_dicthierarchy(d, kw=[]):
    """ Routes the hierarchy of dictionary and returns path and data type information of leaf variables
    
    Args:
        d (dict): a dictionary variable with multi-level hierarchy
    Usage: 
        paths, types = route_dicthierarchy(d:dict)

    Returns:
        paths (list): contains the hierarchical paths(keys) from the root to each leaf, as a list of keywords
        types (list): contains the data types of each leaf value, (e.g.) <class 'str'>, <class 'int'>, <class 'list'>, ...
    """
    paths = []
    types = []
    values= []
    if type(d) != dict:
        return [kw], [type(d)], [d]

    else:
        for i, j in d.items():
            p, t, v = route_dicthierarchy(j, kw + [i])
            paths += p
            types += t
            values+= v
            
    return paths, types, values


def hypload(yaml_path):
    """ Load '.yaml' file
    
    Args:
        yaml_path (str, pathlib.Path): path to .yaml file

    Returns:
        hyps (dict): python dictionary
    """
    with open(yaml_path, 'r') as f:
        hyps = yaml.load(f, Loader=yaml.FullLoader)
            
    return hyps


def hypwrite(config:dict, f_out):
    """ Write '.yaml' file

    Args:
        config (dict): a set of hyperparameter dictionary
        f_out (file object): a file object to write the dictionary, you can also use 'f_out=sys.stdout' to print dict() out
    """
    ruamel_obj  = ruamel.yaml.comments.CommentedMap(config)
    for key in config.keys():
        ruamel_obj.yaml_set_comment_before_after_key(key, before='\n')
    yaml = ruamel.yaml.YAML()
    yaml.indent(mapping=4)
    yaml.dump(ruamel_obj, f_out) 


def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)


def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


def flattenIterable(items):
    """ Yield items from any nested iterable """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flattenIterable(x):
                yield sub_x
        else:
            yield x
            
            
def checkNumParams(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
        