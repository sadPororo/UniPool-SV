""" Utilities to log/print experimental progressions """
import os
import sys
import time
import logging
import neptune

from datetime import datetime

from os.path import join as opj
from utils.utility import route_dicthierarchy, hypwrite


def printlog(stdout:str, logging_path:str=None, is_master:bool=True, silence:bool=False):
    """ Print the string object to sys.stdout and make a log
    
    Args:
        stdout (str): content for stdout.
        logging_path (str): '.log' file path.
        silence (bool, optional): will not print content out to stdout if given True. Defaults to False.
        is_master (bool, optional): only master process will make a log. Defaults to True.
    """
    # print to sys.stdout
    if not silence:
        for line in stdout.replace('\t', ' '*4).split('\n'):
            print(line)
    
    # create a log with the same content (only the master process creates the log)
    if is_master and logging_path is not None:
        logging.basicConfig(filename=logging_path,
                            format='%(asctime)s %(message)s',
                            datefmt='%Y/%m/%d (%I:%M:%S %p)',
                            filemode='a+')
        local_logger = logging.getLogger('train')
        local_logger.setLevel(logging.DEBUG)
        for line in stdout.split('\n'):
            local_logger.debug(line)


def printarg(config, tab_len=4, logging_path:str=None, is_master:bool=True):
    """ Print the current configuration

    Args:
        conf (dict): a python dictionary with a multi-level hierarchy
        tab_len (int, optional): the number of the white spaces distinguishing each level, Defaults to 4.
        logging_path (str, optional): if '.log' file path given, printed content will be logged. Defaults to None.
        is_master (bool, optional): only master process will make a log. Defaults to True.
    """
    def print_dicthierarchy(d, kw=[], lv=0):
        if type(d) != dict:
            if type(d) == str: d = f"'{d}'"
            printlog(tabs*lv + f"%-{max_name_len}s %-{max_type_len}s: {d}"%(f"'{kw[-1]}'", f"({type(d).__name__})"), logging_path, is_master)
            return
        else:
            if lv != 0:
                printlog('', logging_path, is_master)
                printlog(tabs*lv + f"'{kw[-1]}'" + ": {", logging_path, is_master)
            for i, v in d.items():
                print_dicthierarchy(v, kw + [i], lv=lv+1)
            if lv != 0:
                printlog(tabs*(lv+1) + "} ", logging_path, is_master)
            
    paths, types, _ = route_dicthierarchy(config)
    
    max_name_len = max([len(p[-1]) for p in paths]) + 2
    max_type_len = max([len(t.__name__) for t in types]) + 3
    tabs = ' ' * tab_len
    
    printlog('CONFIG: {', logging_path, is_master)
    print_dicthierarchy(config)
    printlog(tabs+'}', logging_path, is_master)


def init_loggings(config:dict, is_master:bool=True):
    """ Initiate the logger objects; offline & neptune (online) logger if configured.
    
    Args:
        config (dict) : hyperparameters
        is_master (bool, optional): only master process will make a log. Defaults to True.

    Returns:
        config (dict) : 'general'-['exp_id', 'result_path', 'logging_path'] key/value has been updated. 
        logger (neptune.run) : logger object for online logging.
    """
    # Only the master process creates the logger
    if is_master:
        # if "--neptune" argument given
        if config['general']['neptune']:
            # Meta tags
            if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
                gpu_device_tag = f"cuda: [{os.environ.get('CUDA_VISIBLE_DEVICES').replace(',', ', ')}]"
            else: gpu_device_tag = 'cuda: %s'%str(config['general']['device'])
            src_ver_tag = os.path.abspath(os.getcwd()).split('/')[-1]
            
            # Get Experiment ID & Initiate Neptune.Run()
            logger = neptune.init_run(project=config['neptune']['project'], 
                                    api_token=config['neptune']['api_token'],
                                    tags=(config['neptune']['location_tag']  # machine location (IP, names)
                                        + [gpu_device_tag]   # gpu-devices
                                        + [src_ver_tag]),    # src version
                                    description=config['general']['description'])
            pths, _, vals = route_dicthierarchy(config)
            logger['parameters'] = dict(dict([('/'.join(k), v) for k, v in zip(pths, vals)]))
            exp_id = str(logger._sys_id)
            
        else:
            logger = None
            exp_id = 'local-' + datetime.now().strftime("%Y%m%d-%H%M%S")
            
        # save the ID of the experiment in './tmp' folder
        with open('./tmp/exp_id.txt', 'w') as f:
            f.write(exp_id)
            
    # the other processes wait until the master initiate the logger
    if not is_master:
        time.sleep(5)
        logger = None
        with open('./tmp/exp_id.txt', 'r') as f:
            exp_id = f.readline()
        exp_id = exp_id.strip()
    
    # master process waits until the others read the expterment ID.
    if is_master:
        time.sleep(10)
        os.remove('./tmp/exp_id.txt')
    
    # Set/Create result path and local-logging path
    config['general']['exp_id']       = exp_id
    config['general']['result_path']  = opj('./res', exp_id)
    config['general']['logging_path'] = opj(config['general']['result_path'], 'training.log')
    os.makedirs(config['general']['result_path'], exist_ok=True)
    
    # Print out the configurations
    if is_master:
        logging_path = config['general']['logging_path']
        
        # print/log the original command line
        printlog('\n\nUSING COMMAND: '+' '.join(sys.argv)+'\n',  logging_path)

        # print/log hyperparameter arguments
        printlog('', logging_path)
        printarg(config, 4, logging_path); 
        printlog('', logging_path)

        # print/log result path and save hyperparameters
        printlog(f"Experiment '{exp_id}' logger created.", logging_path)
        printlog(f"\t- result directory path: '{os.path.abspath(opj(os.getcwd(), config['general']['result_path']))}/'", logging_path)
        printlog(f"\t- hyperparameters saved: '{opj(config['general']['result_path'], 'config.yaml')}'", logging_path)
        with open(opj(config['general']['result_path'], 'config.yaml'), 'w') as f:
            hypwrite(config, f)
        printlog('', logging_path)

        # print/log GPU setup info
        if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
            # device id configured by CUDA_VISIBLE_DEVICE
            cuda_vis_devices = os.environ.get('CUDA_VISIBLE_DEVICES').split(',')
            if len(cuda_vis_devices) > 1:
                printlog('Setting up multi-gpu processing on devices: %s'%str(cuda_vis_devices), logging_path)
            elif len(cuda_vis_devices) == 1:
                printlog('Training will be set on the device: %s'%str(cuda_vis_devices), logging_path)
            else:
                raise ValueError(os.environ.get('CUDA_VISIBLE_DEVICES'))
            printlog(f"\t* CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} passed by the command line, overriding '--device' configuration.")
            
        else:
            # device id configured by passing '--device' argument
            if len(config['general']['device']) > 1:
                printlog('Setting up multi-gpu processing on devices: %s'%str(config['general']['device']), logging_path)
            elif len(config['general']['device'])==1:
                printlog('Training will be set on the device: %s'%str(config['general']['device']), logging_path)
            else:
                printlog('No gpu has configured, will use CPU.',logging_path)
                        
    return config, logger


def resume_loggings(config:dict, is_master:bool=True):
    if is_master and config.get('neptune') is not None:
        logger = neptune.init_run(project=config['neptune']['project'], 
                                  api_token=config['neptune']['api_token'],
                                  with_id=config['general']['exp_id'])
    else:
        logger = None
    
    return logger
    
