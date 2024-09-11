""" A wrapper for the evaluation process """
import os
import sys
import importlib

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from os.path import join as opj
from utils.utility import random_state_init, get_random_state, hypwrite
from utils.loggers import resume_loggings, printlog
from utils.dataset import MulticlassTrainDataset, PairedTrainDataset, EvalDataset, pad_collate_multiclass, pad_collate_binary
from utils.sampler import DistributedEvalSampler
from train import ddp_init



def evaluate(config:dict, device_id:int=0):
    # global setpups
    is_ddp       = dist.is_initialized()
    rank         = 0 if not is_ddp else dist.get_rank()    
    is_master    = rank==0    
    logging_path = config['general']['logging_path']
    # logger = resume_loggings(config, is_master)
    
    # Dataset
    eval_set = EvalDataset(config, mode='test')
    
    # DDP Sampler
    if is_ddp:
        printlog('...initiating samplers for distributed processing.', logging_path, is_master)
        eval_sampler  = DistributedEvalSampler(dataset=eval_set,  shuffle=False)
    else:
        eval_sampler = None
   
    # DataLoader
    eval_loader  = DataLoader(dataset=eval_set, batch_size=1, num_workers=config['general']['workers'],
                              sampler=eval_sampler, shuffle=False, pin_memory=True if is_ddp else False)
    
    # Model
    Model = importlib.import_module('model', package='benchmarks.%s'%config['posarg']['model'])
    model = getattr(Model, 'SVmodel')(config)
    
    model_state_dict = torch.load(opj(config['general']['result_path'], 'model_best.pt'))['model_state']
    for k in list(model_state_dict.keys()):
        if k.startswith('module.'):
            model_state_dict[k.replace('module.', '')] = model_state_dict.pop(k)
    
    model.load_state_dict(model_state_dict)
        
    if is_ddp:
        model = DDP(module=model, device_ids=[device_id], find_unused_parameters=True)
    else:
        model = model.to(device_id)
        
    # Evaluation phase
    val_threshold = torch.load(opj(config['general']['result_path'], 'model_best.pt'))['val_threshold']
            
    Trainer  = importlib.import_module('trainer', package='benchmarks.%s'%config['posarg']['model'])
    eval_time, eval_scr = getattr(Trainer, 'evaluate')(config, model, eval_loader, device_id, threshold=val_threshold)
    
    printlog('\t--------------------------------------------------------------------------------------\n' +
            '\tevaluation time: {:d} min {:02d} sec\n'.format(int(eval_time//60), int(eval_time%60)) +
            "\t'eval'  | EER    : {:.04f}, threshold: {:.03f}\n".format(eval_scr['EER'],    eval_scr['EER_threshold'])  + 
            "\t        | minDCF : {:.04f}, threshold: {:.03f}\n".format(eval_scr['minDCF'], eval_scr['DCF_threshold']) +
            "\t        | FAR*   : {:.04f}, threshold: {:.03f}\n".format(eval_scr['FAR'].item(), val_threshold) +
            "\t        | FRR*   : {:.04f},     -           \n".format(eval_scr['FRR'].item()) +
            "\t        | FRR*   : {:.04f},     -           \n".format((eval_scr['FAR'].item() + eval_scr['FRR'].item()) / 2.), logging_path, is_master)
    
    # # Logging
    # if logger is not None and is_master:
        
    #     logger[f"{config['general']['eval_data']}/EER"].append(eval_scr['EER'])
    #     logger[f"{config['general']['eval_data']}/EER_threshold"].append(eval_scr['EER_threshold'])
        
    #     logger[f"{config['general']['eval_data']}/minDCF"].append(eval_scr['minDCF'])
    #     logger[f"{config['general']['eval_data']}/DCF_threshold"].append(eval_scr['DCF_threshold'])
        
    #     logger[f"{config['general']['eval_data']}/FAR*"].append(eval_scr['FAR'].item())
    #     logger[f"{config['general']['eval_data']}/FRR*"].append(eval_scr['FRR'].item())
    #     logger[f"{config['general']['eval_data']}/EER*"].append((eval_scr['FAR'].item() + eval_scr['FRR'].item()) / 2.)
    #     logger.stop()
        

def ddp_eval(rank:int, config:dict):
    """ main function for evaluate benchmark via DDP """
    
    local_gpu_id, config, _ = ddp_init(rank, config)
    evaluate(config, device_id=local_gpu_id)
    
    dist.destroy_process_group()
    return 0 
    

def evaluate_main(config):
    
    # if config['posarg']['model'] in ['ExploreWV2'] or (config['posarg']['data'] == 'VCTK' and config['general']['exp_id'] in ['SPKID-719', 'SPKID-720', 'SPKID-636','SPKID-721']):
    #     pass
    # else:
    #     print(f"skipping {config['general']['exp_id']}: {config['posarg']['model']}")
    #     sys.exit()
    
    # Print train/test meta configuration
    logging_path = config['general']['logging_path']    
    printlog('\n' + 
             f"  [ EXP-ID: {config['general']['exp_id']} ]\n" +
             'Starting evaluation process...\n' +
             f"\t- model type        : {config['posarg']['model']}\n"
             f"\t- training setup    : {config['posarg']['data']} (trained on {config['data']['nb_class_train']} speakers, {config['data']['train_hrs']} Hrs) \n" +
             f"\t- evaluation trials : {config['general']['eval_data']} (testset comprises {config['data']['nb_class_test']} speakers)\n", logging_path)
    
    # Add package path
    sys.path.append('./src/benchmarks/%s'%config['posarg']['model'])
    
    # Set every random states on devices
    random_state_init(seed=config['general']['seed'], device_list=config['general']['device'])

    # multi GPU processing setup
    if len(config['general']['device']) > 1:
        config['general']['world_size'] = len(config['general']['device'])
        mp.spawn(ddp_eval,
                 args=(config,),
                 nprocs=config['general']['world_size'],
                 join=True)
    
    # single GPU setup
    elif len(config['general']['device']) == 1:
        config['general']['world_size'] = 1
        local_gpu_id = int(config['general']['device'][0])
        if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
            printlog(f"...GPU device (local-id: {local_gpu_id}, global-id: {os.environ.get('CUDA_VISIBLE_DEVICES')}): ready.\n", logging_path)    
        else:
            printlog(f'...GPU device {local_gpu_id}: ready.\n', logging_path)
        evaluate(config, device_id=local_gpu_id)
        
    # else use CPU
    else:
        raise NotImplementedError(config['general']['device'])
        
    
    
    
    
    

