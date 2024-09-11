""" A pytorch reproduction of training strategy referred from the paper, 
[Paper Reference]
- Nik Vaessen, and David A. Van Leeuwen. 
    "Fine-tuning wav2vec2 for speaker recognition." in IEEE ICASSP (2022).
    doi: 10.1109/ICASSP43922.2022.9746952

[Code Reference]
    https://github.com/nikvaessen/w2v2-speaker

[Summary]
- Pretrained Wav2Vec 2.0
    * The usage of pretrained weights can be controlled with '--use_pretrain' argument.

- Optimizer & LR scheduler
    * Adam/SGD optimizer is adopted referring to the Vaessen's github project (code reference).
    * As mentioned in the paper (Sec 4.5., 5.3.), Tri-stage Learning rate scheduling is adopted, and its step-ratio, base-lr, ... are referred from the git above.
    
- Finetuning strategy
    * The argument '--frozen_extractor' will freeze the Convolutional Network from the Wav2Vec 2.0 model for the whole training process, else CNN will be also finetuned.
"""
import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LrScheduler
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import SVmodel
from utils.loggers import printlog
from utils.utility import get_random_state
from utils.metrics import get_Accuracy, get_F1score, get_EER, get_minDCF, get_FAR, get_FRR
from utils.dataset import MulticlassTrainDataset, PairedTrainDataset, EvalDataset, pad_collate_multiclass, pad_collate_binary
from utils.sampler import DistributedEvalSampler
from tristage_func import TriStageLearningRateLambdaLRFunction

from torch.nn.utils import clip_grad_norm_
from os.path import join as opj
from tqdm import tqdm


def train_iteration(config:dict, model:nn.Module, loader:DataLoader, device_id:int, optimizer:optim, scheduler:LrScheduler=None):
    """ 
    Args:
        config (dict): hyperparameters
        model (nn.Module): FinetuneWV2 model
        loader (DataLoader): train loader
        device_id (int): gpu-id to use
        optimizer (optim): Adam
        scheduler (LrScheduler, optional): Tri-stage LR scheduler. Defaults to None.

    Returns:
        EER/minDCF   on 'BCE' setup
        ACC/F1-macro on 'CE' and 'AAM' setups
    """
    # global setups to local FLAG variable
    is_ddp       = dist.is_initialized()
    rank         = 0 if not is_ddp else dist.get_rank()
    is_master    = rank==0
    is_bce       = config['model']['variant']=='BCE'
    is_quickrun  = config['general']['quickrun']
    logging_path = config['general']['logging_path']
    world_size   = config['general']['world_size']
    device_lst   = config['general']['device']    
    
    warmup_stage_steps = scheduler.lr_lambdas[0].warmup_stage_steps
    constant_stage_steps = scheduler.lr_lambdas[0].constant_stage_steps

    # basic training setups
    model.train()
    s_time = time.time()
    batch_loss   = 0.
    train_preds  = torch.empty(0)
    train_labels = torch.empty(0)
    
    # batch-param update iteration
    for i, (waveforms, lengths, labels) in enumerate(tqdm(loader)):
        """
            if 'CE' or 'AAM'
                waveforms : (B, T)
                lengths   : (B,)
                labels    : (B,)
                
            else: 'BCE'
                waveforms : (b, 2, T)
                lengths   : (b, 2)
                labels    : (b,)            
        """
        # Stop iteration in conditions
        if (is_quickrun and i > 100) or config['common']['nb_total_step'] <= scheduler.last_epoch:
            if is_ddp: dist.barrier()
            break
        
        # FLAG notification for scheduler phase
        if scheduler.last_epoch == warmup_stage_steps:
            printlog(f'   ** Training has reached to the end of warm-up phase and set maintain LR in constant (step: {warmup_stage_steps})...', logging_path, is_master)
        if scheduler.last_epoch == warmup_stage_steps + constant_stage_steps:
            printlog(f'   ** Training has reached to the annealing phase, will decay LR exponentially (step: {warmup_stage_steps + constant_stage_steps})...', logging_path, is_master)
        
        # minibatch to cuda device
        if is_bce:
            waveforms = waveforms.reshape(waveforms.size(0)*2, waveforms.size(-1)).to(device_id) # (2xb, T)
            lengths   = lengths.reshape(lengths.size(0)*2).to(device_id) # (2xb)
            labels    = labels.float().to(device_id) # (b)
        else:
            waveforms = waveforms.to(device_id) # (B, T)
            lengths   = lengths.to(device_id) # (B,)
            labels    = labels.to(device_id)  # (B,)
        
        # model forward
        pred, loss = model(waveforms, lengths, labels)
        
        # backward with update
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.)
        optimizer.step()
        scheduler.step()
        
        # accumualte stats
        batch_loss  += loss.item()
        train_labels = torch.cat([train_labels, labels.detach().cpu()], dim=0)
        if is_bce:
            train_preds  = torch.cat([train_preds, pred.detach().cpu()], dim=0)
        else:
            train_preds  = torch.cat([train_preds, pred.argmax(dim=-1).detach().cpu()], dim=0)
            
    # if DDP, collect results from each subprocess (rank)
    if is_ddp:
        torch.save((batch_loss, train_preds, train_labels), f'./tmp/part_rank{rank}_device{device_id}.pt')
        dist.barrier()
        
        if is_master:
            batch_loss   = 0.
            train_preds  = torch.empty(0)
            train_labels = torch.empty(0)
            for r, d in zip(range(world_size), device_lst):
                (part_loss, part_preds, part_labels) = torch.load(f'./tmp/part_rank{r}_device{d}.pt')
                batch_loss  += part_loss
                train_preds  = torch.cat([train_preds,  part_preds], dim=0)
                train_labels = torch.cat([train_labels, part_labels], dim=0)
                os.remove(f'./tmp/part_rank{r}_device{d}.pt')
            del part_loss, part_preds, part_labels

    # score metrics
    train_scr = {
        'train_loss': 99., 'ACC': 0., 'F1': 0.,
        'EER'   : 100., 'EER_threshold' : 100.,
        'minDCF': 100., 'DCF_threshold' : 100.,
    }
    if is_master:
        train_scr['train_loss'] = batch_loss / len(loader)
        
        if is_bce:
            train_scr['EER'], train_scr['EER_threshold'] = get_EER(train_labels, train_preds)
            
            if len(train_labels) >= 50000:
                # 100K samples cost ~80Gb Memory to calculate minDCF
                # 50K  samples cost 35~40Gb Memory
                DCF_list = [[], []] # [0] for minDCF scores, [1] for the thresholds
                
                N_sample = len(train_labels) // 50000 + int(bool(len(train_labels) % 50000))
                if N_sample > 10:
                    N_sample = 10

                for _ in range(N_sample):
                    sample_indice = random.sample(torch.arange(len(train_labels)).tolist(), 50000)
                    minDCF, DCF_threshold = get_minDCF(train_labels[sample_indice], train_preds[sample_indice])
                    
                    DCF_list[0].append(minDCF)
                    DCF_list[1].append(DCF_threshold)
                
                train_scr['minDCF'],     train_scr['DCF_threshold']     = np.mean(DCF_list[0]), np.mean(DCF_list[1])
                train_scr['minDCF_std'], train_scr['DCF_threshold_std'] = np.std(DCF_list[0]),  np.std(DCF_list[1]) 
            
            else:
                train_scr['minDCF'], train_scr['DCF_threshold'] = get_minDCF(train_labels, train_preds)
                         
        else:
            train_scr['ACC'] = get_Accuracy(train_labels, train_preds)
            train_scr['F1']  = get_F1score(train_labels, train_preds)

    torch.cuda.empty_cache()
    if is_ddp: dist.barrier()
    train_time = time.time() - s_time

    return train_time, train_scr


def evaluate(config:dict, model:nn.Module, loader:DataLoader, device_id:int, threshold=None):
    """
    Args:
        config (dict): hyperparameters
        model (nn.Module): ExploreWV model
        loader (DataLoader): train loader
        device_id (int): gpu-id to use
        rank (optional, int): local rank of this process. Defaults to 0.
        is_ddp (optional, bool): a flag for distributed processing. Defaults to False.         
    """
    # global setups to local FLAG variables    
    is_ddp      = dist.is_initialized()
    rank        = 0 if not is_ddp else dist.get_rank()    
    is_master   = rank==0
    is_bce      = config['model']['variant']=='BCE'
    is_quickrun = config['general']['quickrun']
    world_size  = config['general']['world_size']
    device_lst  = config['general']['device']

    # basic eval setups
    model.eval()
    s_time = time.time()
    eval_simils = []
    eval_labels = []
    
    # eval batch iteration
    with torch.no_grad():
        for i, (waveform0, waveform1, label) in enumerate(tqdm(loader)):
            
            if is_quickrun and i > 100:
                break
            
            if is_bce:
                sim_scr = model([waveform0.to(device_id), 
                                 waveform1.to(device_id)]).detach().cpu().item()
                
            else: # CE | AAM
                x_vec0 = model(waveform0.to(device_id))
                x_vec1 = model(waveform1.to(device_id))
                
                sim_scr = torch.matmul(x_vec0, x_vec1.T).clamp(0, 1).detach().cpu().item()
                # sim_scr = torch.matmul(x_vec0, x_vec1.T).detach().cpu().item() * .5 + .5

            # accumulate stats
            eval_simils.append(sim_scr)
            eval_labels.append(label.item())
        
        eval_simils = torch.Tensor(eval_simils)
        eval_labels = torch.LongTensor(eval_labels)
            
    # if DDP, collect results from each part (rank)
    if is_ddp:
        torch.save((eval_simils, eval_labels), f'./tmp/part_rank{rank}_device{device_id}.pt')
        dist.barrier()
        
        if is_master:
            eval_simils = torch.empty(0)
            eval_labels = torch.empty(0)
            for r, d in zip(range(world_size), device_lst):
                (part_simils, part_labels) = torch.load(f'./tmp/part_rank{r}_device{d}.pt')
                eval_simils = torch.cat([eval_simils, part_simils], dim=0)
                eval_labels = torch.cat([eval_labels, part_labels], dim=0)
                os.remove(f'./tmp/part_rank{r}_device{d}.pt')
            del part_simils, part_labels
    
    # score metrics
    eval_scr = {'EER' : 999., 'EER_threshold': 100, 'minDCF': 999., 'DCF_threshold': 100,
                'FAR' : torch.Tensor([999.]), 'FAR_threshold': torch.Tensor([100.]), 
                'FRR' : torch.Tensor([999.]), 'FRR_threshold': torch.Tensor([100.])}
     
    if is_master:
        eval_scr['EER'],    eval_scr['EER_threshold'] = get_EER(eval_labels, eval_simils)
        eval_scr['minDCF'], eval_scr['DCF_threshold'] = get_minDCF(eval_labels, eval_simils)
        
        eval_scr['FAR'], eval_scr['FAR_threshold'] = get_FAR(eval_labels, eval_simils, threshold)
        eval_scr['FRR'], eval_scr['FRR_threshold'] = get_FRR(eval_labels, eval_simils, threshold)

    torch.cuda.empty_cache()
    if is_ddp: dist.barrier()
    eval_time = time.time() - s_time

    return eval_time, eval_scr


def train(config:dict, logger=None, device_id:int=0):
    """ This function reproduces the speaker verification model training introduced from the paper,
    - Nik Vaessen, and David A. Van Leeuwen. "Fine-tuning wav2vec2 for speaker recognition." in IEEE ICASSP (2022).
        doi: 10.1109/ICASSP43922.2022.9746952

    Args:
        config (dict): hyperparams
        logger (optional, Neptune.run): Neptune logger object, if configured. Defaults to None
        device_id (optional, int): device id to use. Defaults to 0.
    """
    # global setups to local FLAG variables    
    is_ddp       = dist.is_initialized()
    rank         = 0 if not is_ddp else dist.get_rank()    
    is_master    = rank==0    
    is_bce       = config['model']['variant']=='BCE'
    is_quickrun  = config['general']['quickrun']
    is_skiptest  = config['general']['skiptest']
    is_neptune   = config['general']['neptune']
    logging_path = config['general']['logging_path']
    
    # Dataset
    printlog('...initiating datasets.', logging_path, is_master)
    if is_bce:
        printlog("\t- model variant 'BCE' is selected, will override 'config:model:bsz_ratio' to 1.", logging_path, is_master)    
        train_set = PairedTrainDataset(config)
        config['model']['bsz_ratio'] = 1
    else:
        train_set = MulticlassTrainDataset(config)
    valid_set = EvalDataset(config, mode='valid')
    test_set  = EvalDataset(config, mode='test')
    
    # DDP Sampler
    if is_ddp:
        printlog('...initiating samplers for distributed processing.', logging_path, is_master)
        train_sampler = DistributedSampler(dataset=train_set, shuffle=True)
        valid_sampler = DistributedEvalSampler(dataset=valid_set, shuffle=False)
        test_sampler  = DistributedEvalSampler(dataset=test_set,  shuffle=False)
    else:
        train_sampler, valid_sampler, test_sampler = None, None, None
    
    # DataLoaders    
    if is_ddp:
        world_size     = config['general']['world_size']
        trn_batch_size = config['common']['batch_size'] // config['general']['world_size']
        evl_batch_size = 1
        num_workers    = config['general']['workers']
        trn_shuffle    = False
        pin_memory     = True
        printlog('...setting dataloaders for DDP.\n' +
                 f'\t- total batch size : {trn_batch_size*world_size} ({trn_batch_size} per device) | on evaluation: {world_size} (fixed to 1 per device): \n' +
                 f'\t- total num workers: {num_workers*world_size}', logging_path, is_master)
    else:
        trn_batch_size = config['common']['batch_size']
        evl_batch_size = 1
        num_workers    = config['general']['workers']
        trn_shuffle    = True
        pin_memory     = False
        printlog('...setting dataloaders.\n' +
                 f'\t- batch size : {trn_batch_size} | fixed to 1 on evaluation\n' +
                 f'\t- num workers: {num_workers}', logging_path, is_master)
    
    if is_bce:
        train_loader = DataLoader(dataset=train_set, batch_size=trn_batch_size, num_workers=num_workers, 
                                  sampler=train_sampler, shuffle=trn_shuffle, pin_memory=pin_memory, collate_fn=pad_collate_binary)
    else:
        train_loader = DataLoader(dataset=train_set, batch_size=trn_batch_size, num_workers=num_workers, 
                                  sampler=train_sampler, shuffle=trn_shuffle, pin_memory=pin_memory, collate_fn=pad_collate_multiclass)
    valid_loader = DataLoader(dataset=valid_set, batch_size=evl_batch_size, num_workers=num_workers, 
                              sampler=valid_sampler, shuffle=False, pin_memory=pin_memory)
    test_loader  = DataLoader(dataset=test_set, batch_size=evl_batch_size, num_workers=num_workers,
                              sampler=test_sampler, shuffle=False, pin_memory=pin_memory)

    # Network model
    printlog('...initiating the network model.', logging_path, is_master)
    model = SVmodel(config)
    model = model.cuda(device_id)
    if is_ddp:
        model = DDP(module=model, device_ids=[device_id], find_unused_parameters=True)
        w2v_feature_extractor_params = model.module.w2v_model.feature_extractor.parameters()        
    else:
        w2v_feature_extractor_params = model.w2v_model.feature_extractor.parameters()
        
    # Freeze Wav2Vec 2.0 - Feature extractor (Convolution Network) if configured
    if config['model']['use_pretrain'] and config['model']['frozen_extractor']:
        for p in w2v_feature_extractor_params:
            p.requires_grad_(False)
        if is_ddp: dist.barrier()
        printlog(f"\t- FLAG: 'frozen_extractor' is set 'True', Convolution layers from the Wav2Vec 2.0 will remain frozen.", logging_path, is_master)
    
    # Initiate optimizer/scheduler
    printlog('...setting optimizer/scheduler.', logging_path, is_master)
    printlog('\t- optimizer/scheduler parameters and the ratios are referred from,', logging_path, is_master)
    printlog('\t\t* https://github.com/nikvaessen/w2v2-speaker/tree/master/config/optim', logging_path, is_master)
    printlog('\t\t* doi: 10.1109/ICASSP43922.2022.9746952', logging_path, is_master)
    
    # Base learning rate setup
    if config['model']['base_lr'] != 0:
        base_lr = config['model']['base_lr']
        printlog(f"\t- manual 'base_lr' is configured to {base_lr}.", logging_path, is_master)    
    else:
        if config['model']['variant'] == 'CE': base_lr = 9e-5
        elif config['model']['variant'] == 'AAM': base_lr = 5e-5
        elif config['model']['variant'] == 'BCE': base_lr = 3e-5
        printlog(f"\t- default 'base_lr' is set for the variant '{config['model']['variant']}': {base_lr}", logging_path, is_master)

    # Optimizer setup
    if config['model']['optimizer'] == 'Adam':
        optimizer = optim.Adam(params=model.parameters(), 
                               lr=base_lr, weight_decay=0, eps=1e-8, amsgrad=False)
        printlog(f'\t- using optimizer/scheduler:\n' + 
                 f'\t\t* Adam (lr: {base_lr}, weight_decay: 0, betas:[0.9, 0.999], eps: 1e-8, amsgrad: False)', logging_path, is_master)
    elif config['model']['optimizer'] == 'SGD':
        optimizer = optim.SGD(params=model.parameters(), 
                              lr=base_lr, momentum=0.9, weight_decay=0.0, dampening=0, nesterov=True)
        printlog(f'\t- using optimizer/scheduler:\n' + 
                 f'\t\t* SGD (lr: {base_lr}, momentum: 0.9, weight_decay: 0, dampening: 0, nesterov: True)', logging_path, is_master)

    # Scheduler setup
    total_steps = int(config['common']['nb_total_step'])
    nb_evals    = int(np.ceil(config['common']['nb_total_step'] / config['common']['nb_steps_eval']))
    warmup_end  = int(total_steps * config['model']['warmup_stage_ratio'])
    const_end   = int(total_steps * (config['model']['warmup_stage_ratio'] + config['model']['constant_stage_ratio']))
    scheduler   = LrScheduler.LambdaLR(optimizer=optimizer,
                                       lr_lambda=TriStageLearningRateLambdaLRFunction(
                                           max_steps=total_steps,
                                           warmup_stage_ratio = config['model']['warmup_stage_ratio'],
                                           constant_stage_ratio = config['model']['constant_stage_ratio'],
                                           decay_stage_ratio = config['model']['decay_stage_ratio'],
                                           initial_lr=config['model']['initial_lr'],
                                           base_lr=base_lr,
                                           final_lr=config['model']['final_lr'],
                                          )
                                       )
    printlog(f'\t\t* TriStageLRscheduler (\n' +
             f"\t\t\t  max_steps: {total_steps}\n" +
             f"\t\t\t  warmup_stage_ratio: {config['model']['warmup_stage_ratio']}   -> (warm-up linearly for the first {warmup_end} steps,\n" +
             f"\t\t\t  constant_stage_ratio: {config['model']['constant_stage_ratio']} ->  and set lr constant until the {const_end}_th step,\n" +
             f"\t\t\t  decay_stage_ratio: {config['model']['decay_stage_ratio']}    ->  then exponential decrease to the end, total {total_steps} steps)\n" +
             f"\t\t\t  initial_lr: {config['model']['initial_lr']}\n" +
             f"\t\t\t  base_lr: {base_lr}\n" +
             f"\t\t\t  final_lr: {config['model']['final_lr']}\n" +
             f'\t\t  )', logging_path, is_master)

    # Calculate epoch approximation
    train_size    = train_set.get_total_dataset_size()
    train_hrs     = config['data']['train_hrs']
    batch_size    = config['common']['batch_size']
    train_max_sec = config['common']['train_max_sec']
    train_avg_dur = config['data']['train_hrs'] * 60 * 60 / train_set.get_total_dataset_size()    
    printlog(f'...Quantitative measurement of configured number of iterations via epoch unit.\n' \
             f"\t- Training set comprises {train_size} audio samples for {train_hrs} Hrs in total, {round(train_avg_dur, 2)} seconds duration per sample on average.", logging_path, is_master)
    if train_max_sec > train_avg_dur: # the average duration matters more than maximum duration
        printlog(f"\t\t* train average-duration: {round(train_avg_dur, 2)} < max-duration: {train_max_sec},\n" +
                 f"\t\t  the AVG-duration matters more than the maximum in training.\n" +
                 f"\t- ({total_steps}) steps x ({batch_size}) batch x ({round(train_avg_dur, 2)}) second / ({train_hrs}) hrs" , logging_path, is_master)
        apprx_nb_epoch = (total_steps * batch_size * train_avg_dur) / (train_hrs * 60 * 60)
    else:
        printlog(f"\t\t* train max-duration: {train_max_sec} <= average-duration: {round(train_avg_dur, 2)},\n" +
                 f"\t\t  the MAX-duration matters more than the average in training.\n" +
                 f"\t- ({total_steps}) steps x ({batch_size}) batch x ({train_max_sec}) second / ({train_hrs}) hrs" , logging_path, is_master)
        apprx_nb_epoch = (total_steps * batch_size * train_max_sec) / (train_hrs * 60 * 60)        
    printlog(f"\t\t* approximately ({round(apprx_nb_epoch, 2)}) epochs are considered.\n", logging_path, is_master)
        
    # Initiate the meta data for saving the best model
    start_eval_no = 1
    last_eval_no  = 3 if is_quickrun else nb_evals
    best_valid_scr = {'EER':9999., 'EER_threshold':1000., 'minDCF':9999., 'DCF_threshold':1000.}
    best_test_scr  = {'EER':9999., 'EER_threshold':1000., 'minDCF':9999., 'DCF_threshold':1000., 
                      'FAR*':9999., 'FRR*':9999., 'EER*':9999.}
    
    # Training epoch starts here
    printlog('Model training iteration start...', logging_path, is_master)
    for cur_eval_no in range(start_eval_no, last_eval_no+1):
        # Training Phase
        printlog('Iteration no.{:03d}/{:03d} (step: {:d} to {:d} / total {:d})...'.format(cur_eval_no, last_eval_no, 
                                                                                          (cur_eval_no-1)*len(train_loader)+1,
                                                                                          min(total_steps, (cur_eval_no)*len(train_loader)),
                                                                                          total_steps), logging_path, is_master)
        if cur_eval_no > 1:
            train_set.sample_for_iteration()        
        if train_sampler is not None:
            train_sampler.set_epoch(cur_eval_no)
            
        if is_ddp: dist.barrier()
        train_time, train_scr = train_iteration(config, model, train_loader, device_id, optimizer, scheduler)
        
        # Evaluation Phase
        valid_time, valid_scr = evaluate(config, model, valid_loader, device_id)
        if is_skiptest:
            test_time, test_scr = 0., best_test_scr
        else:
            test_time, test_scr = evaluate(config, model, test_loader, device_id, threshold=valid_scr['EER_threshold'] if is_master else None)
                
        # Save the best validated model when the best score has been updated
        if best_valid_scr['EER'] >= valid_scr['EER']:
            if is_master:
                torch.save({
                    'epoch_no': scheduler.last_epoch,
                    'rand_state': get_random_state(config['general']['device']),
                    'optimizer_state':optimizer.state_dict(),
                    'scheduler_state':scheduler.state_dict(),
                        'model_state': model.module.state_dict() if is_ddp else model.state_dict(),
                        'val_FAR': valid_scr['FAR'],
                        'val_FRR': valid_scr['FRR'],
                        'val_FAR/FRR_thresh': valid_scr['FAR_threshold'],
                        'val_threshold': valid_scr['EER_threshold'],
                }, opj(config['general']['result_path'], 'model_best.pt'))
                
            best_valid_scr['EER'] = valid_scr['EER']
            best_valid_scr['EER_threshold'] = valid_scr['EER_threshold']
            best_valid_scr['minDCF'] = valid_scr['minDCF']
            best_valid_scr['DCF_threshold'] = valid_scr['DCF_threshold']
            
            best_test_scr['EER'] = test_scr['EER']
            best_test_scr['EER_threshold'] = test_scr['EER_threshold']
            best_test_scr['minDCF'] = test_scr['minDCF']
            best_test_scr['DCF_threshold'] = test_scr['DCF_threshold']
            
            best_test_scr['FAR*'] = test_scr['FAR'].item()
            best_test_scr['FRR*'] = test_scr['FRR'].item()
            best_test_scr['EER*'] = (test_scr['FAR'].item() + test_scr['FRR'].item()) / 2.
            
            
        # Save model checkpoint for resuming
        if is_master:
            torch.save({
                'epoch_no': scheduler.last_epoch,
                'rand_state': get_random_state(config['general']['device']),
                'optimizer_state':optimizer.state_dict(),
                'scheduler_state':scheduler.state_dict(),
                'model_state': model.module.state_dict() if is_ddp else model.state_dict()
            }, opj(config['general']['result_path'], 'model_ckpt.pt'))
           
        # print the training results out
        printlog('\ttraining time: {:d} min {:02d} sec'.format(int(train_time//60), int(train_time%60)), logging_path, is_master)
        if is_bce:
            if 'minDCF_std' in train_scr: # train predictions are sampled to calculate minDCF
                printlog("\t'train' | EER: {:.04f}, minDCF: {:.04f} ±{:.04f}, loss: {:.04f}".format(train_scr['EER'], train_scr['minDCF'], train_scr['minDCF_std'], train_scr['train_loss']), logging_path, is_master)                
            else:
                printlog("\t'train' | EER: {:.04f}, minDCF: {:.04f}, loss: {:.04f}".format(train_scr['EER'], train_scr['minDCF'], train_scr['train_loss']), logging_path, is_master)
        else: 
            # AAM & CE
            printlog("\t'train' | ACC: {:.02f}%, F1-macro: {:.02f}%, loss: {:.04f}".format(train_scr['ACC']*100, train_scr['F1']*100, train_scr['train_loss']), logging_path, is_master)
            
        # print the evaluation results out
        printlog('\t--------------------------------------------------------------------------------------\n' + 
                 '\tvalidation time: {:d} min {:02d} sec\n'.format(int(valid_time//60), int(valid_time%60)) +
                 "\t'valid' | EER:    {:.04f}, threshold: {:.03f}".format(valid_scr['EER'], valid_scr['EER_threshold'])    + " | * 'best' EER:    {:.04f}, threshold: {:.03f}\n".format(best_valid_scr['EER'], best_valid_scr['EER_threshold']) + 
                 "\t        | minDCF: {:.04f}, threshold: {:.03f}".format(valid_scr['minDCF'], valid_scr['DCF_threshold']) + " |          minDCF: {:.04f}, threshold: {:.03f}".format(best_valid_scr['minDCF'], best_valid_scr['DCF_threshold']), logging_path, is_master)
        if is_skiptest:
            printlog('\t--------------------------------------------------------------------------------------\n' +
                    "\t- FLAG: '--skiptest' is True, skipping test evaluation.\n", logging_path, is_master)
        else:
            printlog('\t--------------------------------------------------------------------------------------\n' +
                    '\ttesting time: {:d} min {:02d} sec\n'.format(int(test_time//60), int(test_time%60)) +
                    "\t'test'  | EER:    {:.04f}, threshold: {:.03f}".format(test_scr['EER'], test_scr['EER_threshold'])         + " |          EER:    {:.04f}, threshold: {:.03f}\n".format(best_test_scr['EER'], best_test_scr['EER_threshold']) + 
                    "\t        | minDCF: {:.04f}, threshold: {:.03f}".format(test_scr['minDCF'], test_scr['DCF_threshold'])      + " |          minDCF: {:.04f}, threshold: {:.03f}\n".format(best_valid_scr['minDCF'], best_valid_scr['DCF_threshold']) +
                    "\t        | FAR*:   {:.04f}, threshold: {:.03f}".format(test_scr['FAR'].item(), valid_scr['EER_threshold']) + " |          FAR*:   {:.04f}, threshold: {:.03f}\n".format(best_test_scr['FAR*'], best_test_scr['EER_threshold']) + 
                    "\t        | FRR*:   {:.04f},     -           ".format(test_scr['FRR'].item())                               + " |          FRR*:   {:.04f},     -            \n".format(best_test_scr['FRR*']) + 
                    "\t        | EER*:   {:.04f},     -           ".format((test_scr['FAR']+test_scr['FRR']).item() / 2.)        + " |          EER*:   {:.04f},     -            \n".format(best_test_scr['EER*']), logging_path, is_master)
                        
        # Make an online log if specified
        if is_neptune and logger is not None and is_master:
            # cur_step = round((min(total_steps, (cur_eval_no)*len(train_loader)) / total_steps) * apprx_nb_epoch, 2)
            cur_step = scheduler.last_epoch
            
            logger['train/running/loss'].append(train_scr['train_loss'], step=cur_step)
            if is_bce:
                logger['train/running/EER'].append(train_scr['EER'], step=cur_step)
                logger['train/running/minDCF'].append(train_scr['minDCF'], step=cur_step)
            else:
                logger['train/running/acc'].append(train_scr['ACC'], step=cur_step)
                logger['train/running/f1'].append(train_scr['F1'], step=cur_step)
            
            # log the running scores
            logger['valid/running/EER'].append(valid_scr['EER'], step=cur_step)
            logger['valid/running/EER_threshold'].append(valid_scr['EER_threshold'], step=cur_step)
            logger['valid/running/minDCF'].append(valid_scr['minDCF'], step=cur_step)
            logger['valid/running/DCF_threshold'].append(valid_scr['DCF_threshold'], step=cur_step)
            
            logger['test/running/EER'].append(test_scr['EER'], step=cur_step)
            logger['test/running/threshold'].append(test_scr['EER_threshold'], step=cur_step)
            logger['test/running/minDCF'].append(test_scr['minDCF'], step=cur_step)
            logger['test/running/DCF_threshold'].append(test_scr['DCF_threshold'], step=cur_step)
            
            logger['test/running/FAR*'].append(test_scr['FAR'].item(), step=cur_step)
            logger['test/running/FRR*'].append(test_scr['FRR'].item(), step=cur_step)
            logger['test/running/EER*'].append((test_scr['FAR']+test_scr['FRR']).item() / 2., step=cur_step)
            
            
            # track the best scores
            logger['valid/eval/EER'].append(best_valid_scr['EER'], step=cur_step)
            logger['valid/eval/EER_threshold'].append(best_valid_scr['EER_threshold'], step=cur_step)
            logger['valid/eval/minDCF'].append(best_valid_scr['minDCF'], step=cur_step)
            logger['valid/eval/DCF_threshold'].append(best_valid_scr['DCF_threshold'], step=cur_step)
            
            logger['test/eval/EER'].append(best_test_scr['EER'], step=cur_step)
            logger['test/eval/EER_threshold'].append(best_test_scr['EER_threshold'], step=cur_step)
            logger['test/eval/minDCF'].append(best_test_scr['minDCF'], step=cur_step)
            logger['test/eval/DCF_threshold'].append(best_test_scr['DCF_threshold'], step=cur_step)
            
            logger['test/eval/FAR*'].append(best_test_scr['FAR*'], step=cur_step)
            logger['test/eval/FRR*'].append(best_test_scr['FRR*'], step=cur_step)
            logger['test/eval/EER*'].append(best_test_scr['EER*'], step=cur_step)
            
        
        # Sync the processes before to start the next epoch round
        if is_ddp: dist.barrier()
        
    # Detroy online logger if specified
    if is_neptune and logger is not None and is_master:
        logger.stop()

    pass
