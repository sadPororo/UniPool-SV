""" This is a reproduction of the training procedure referred from the paper,
[Paper Reference]
- Fan, Zhiyun, Meng Li, Shiyu Zhou, and Bo Xu. 
    "Exploring wav2vec 2.0 on speaker verification and language identification." in Interspeech (2021).
    doi: 10.21437/Interspeech.2021-1280
    
[Summary]
- Pretrained Wav2Vec 2.0
    * The usage of pretrained weights can be controlled with '--use_pretrain' argument.
  
- Zero-shot evaluation
    * available by passing "--zshot_eval" argument at the main command line.
  
- Optimizer & LR scheduler
    * Adam / OneCycleLR is adopted with the same configuration and ratio of scheduling steps mentioned from the paper.
    * The strategy to freeze/unfreeze the pretrained weights of Wav2Vec 2.0 model from a certain phase. (Transformer layers)
"""
import os
import time
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
from utils.dataset import MulticlassTrainDataset, EvalDataset, pad_collate_multiclass, pad_collate_binary
from utils.sampler import DistributedEvalSampler

from torch.nn.utils import clip_grad_norm_
from os.path import join as opj
from tqdm import tqdm


def train_iteration(FROZEN_WV:bool,  w2v_unfrz_step:int, 
                    config:dict, model:nn.Module, loader:DataLoader, device_id:int, optimizer:optim, scheduler:LrScheduler=None):
    """
    Args:
        FROZEN_WV (bool): a flag to check if w2v-encoder is already unfrozen. True indicates w2v is currently frozen
        w2v_unfrz_step (int): a threshold to unfreeze Wav2Vec 2.0 transformer layer
        config (dict): hyperparameters
        model (nn.Module): ExploreWV2 model
        loader (DataLoader): train loader
        device_id (int): gpu-id to use
        optimizer (optim): Adam optimizer
        scheduler (LrScheduler, optional): OneCycleLR. Defaults to None.

    Returns:
        FROZEN_WV (bool): a flag to check if w2v-encoder is already unfrozen.
        train_time (float): time spent for training (in seconds)
        train_scr (dict): score results of the training iteration
    """
    # global setups to local FLAG variables
    is_ddp        = dist.is_initialized()
    rank          = 0 if not is_ddp else dist.get_rank()
    is_master     = rank==0
    is_quickrun   = config['general']['quickrun']
    logging_path  = config['general']['logging_path']
    world_size    = config['general']['world_size']
    device_lst    = config['general']['device']
    nb_total_step = config['common']['nb_total_step']
    
    # basic training setups
    model.train()
    s_time = time.time()
    batch_loss   = 0.
    train_preds  = torch.empty(0)
    train_labels = torch.empty(0)

    # the batch-param update iteration
    for i, (waveforms, lengths, labels) in enumerate(tqdm(loader)):
        """
            waveforms : (B, T)
            lengths   : (B,)
            labels    : (B,)
        """
        # Stop iteration in conditions
        if (is_quickrun and i > 100) or nb_total_step <= scheduler.last_epoch:
            if is_ddp: dist.barrier()
            break
        
        # Unfreeze w2v-encoder from a certain phase
        if FROZEN_WV and (scheduler.last_epoch >= w2v_unfrz_step):
            printlog(f'   ** Training has reached the step for unfreezing the W2V-encoder (step: {w2v_unfrz_step})...', logging_path, is_master)
            if is_ddp:
                w2v_context_net = model.module.w2v_encoder.encoder.transformer
            else:
                w2v_context_net = model.w2v_encoder.encoder.transformer
            for name, p in w2v_context_net.named_parameters():
                if 'layers' not in name:
                    p.requires_grad_(True)
                else:
                    if int(name.split('.')[1]) < config['model']['num_layers']:
                        p.requires_grad_(True)
                    else: continue            
            FROZEN_WV = False
            if is_ddp: dist.barrier()
            
        # minibatch to cuda device
        waveforms = waveforms.to(device_id)
        lengths   = lengths.to(device_id)
        labels    = labels.to(device_id)
        
        # model forward
        logit, loss = model(waveforms, lengths, labels)
        
        # backward with update
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.)
        optimizer.step()
        scheduler.step()

        # accumulate stats
        batch_loss  += loss.item()
        class_pred   = logit.argmax(dim=-1)
        train_preds  = torch.cat([train_preds,  class_pred.detach().cpu()], dim=0)
        train_labels = torch.cat([train_labels, labels.detach().cpu()], dim=0)
        
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
        'loss': 99.,
        'ACC': 0.,
        'F1':  0.
    }
    if is_master:
        train_scr['loss'] = batch_loss / len(loader)
        train_scr['ACC']  = get_Accuracy(train_labels, train_preds)
        train_scr['F1']   = get_F1score(train_labels, train_preds)        

    torch.cuda.empty_cache()
    if is_ddp: dist.barrier()
    train_time = time.time() - s_time

    return FROZEN_WV, train_time, train_scr


def evaluate(config:dict, model:nn.Module, loader:DataLoader, device_id:int, threshold=None):
    """
    Args:
        config (dict): hyperparameters
        model (nn.Module): ExploreWV2 model
        loader (DataLoader): train loader
        device_id (int): gpu-id to use
    
    Returns:
        eval_time (float): time spent for the evaluation (in seconds)
        eval_scr (dict): score results of model for the given evaluation samples
    """
    # global setups to local FLAG variables
    is_ddp      = dist.is_initialized()
    rank        = 0 if not is_ddp else dist.get_rank()    
    is_master   = rank==0
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
            # Stop iteration in conditions
            if is_quickrun and i > 100:
                break
            
            x_vec0 = model(waveform0.to(device_id), lengths=torch.LongTensor([len(waveform0[0])]).to(device_id))
            x_vec1 = model(waveform1.to(device_id), lengths=torch.LongTensor([len(waveform1[0])]).to(device_id))
            
            sim_scr = torch.matmul(x_vec0, x_vec1.T).clamp(0, 1).detach().cpu().item()
            # sim_scr = torch.matmul(x_vec0, x_vec1.T).detach().cpu().item() * .5 + .5
            
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
    """ The function reproduces the speaker verification training framework referring to the paper,
    - Fan, Zhiyun, Meng Li, Shiyu Zhou, and Bo Xu. "Exploring wav2vec 2.0 on speaker verification and language identification." in Interspeech (2021).
        doi: 10.21437/Interspeech.2021-1280

    Args:
        config (dict): hyperparams
        logger (optional, Neptune.run): Neptune logger object, if configured. Defaults to None
        device_id (optional, int): device id to use. Defaults to 0.
    """
    # global setups to local FLAG variables    
    is_ddp       = dist.is_initialized()
    rank         = 0 if not is_ddp else dist.get_rank()    
    is_master    = rank==0
    is_quickrun  = config['general']['quickrun']
    is_skiptest  = config['general']['skiptest']
    is_neptune   = config['general']['neptune']
    logging_path = config['general']['logging_path']
    
    # Dataset
    printlog('...initiating datasets.', logging_path, is_master)
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
        w2v_params = model.module.w2v_encoder.parameters()        
    else:
        w2v_params = model.w2v_encoder.parameters()
        
    # Freeze Wav2Vec 2.0 weights first if using pretrained
    if config['model']['use_pretrain']:
        for p in w2v_params:
            p.requires_grad_(False)

    # We adopt to copy the ratio of steps due to using diverse datasets
    printlog('...setting optimizer/scheduler.\n' \
             '\t- optimizer/scheduler parameters and the schedule ratios are referred from,\n' \
             '\t\t* doi:10.21437/Interspeech.2021-1280', logging_path, is_master)
    total_steps     = int(config['common']['nb_total_step'])
    nb_evals        = int(np.ceil(config['common']['nb_total_step'] / config['common']['nb_steps_eval']))
    warmup_steps    = int(total_steps * (config['model']['warmup_steps'] / config['model']['total_steps']))
    w2v_unfrz_step  = int(total_steps * (config['model']['w2v_freeze_steps'] / config['model']['total_steps'])) if config['model']['use_pretrain'] else 0

    # Initiate optimizer/scheduler
    optimizer = optim.Adam(params=model.parameters())
    scheduler = LrScheduler.OneCycleLR(optimizer=optimizer,
                                       max_lr          = config['model']['max_lr'],
                                       total_steps     = total_steps,
                                       pct_start       = config['model']['warmup_steps'] / config['model']['total_steps'], 
                                       anneal_strategy = config['model']['anneal_strategy'])
    printlog(f'\t- using optimizer: Adam()\n' \
             f'\t        scheduler: \n' \
             f'\t           OneCycleLR (\n' \
             f"\t               max_lr: {config['model']['max_lr']}\n" \
             f"\t               total_steps: {total_steps}\n" \
             f"\t               pct_start: {round(config['model']['warmup_steps'] / config['model']['total_steps'], 2)}... " \
             f"     -> (original paper: {config['model']['warmup_steps']} warmup steps / {config['model']['total_steps']} in total)\n" \
             f"\t               anneal_strategy: {config['model']['anneal_strategy']} -> (warm-up until : {warmup_steps}, then 'lr' decays linearly)\n" \
             f"\t           )", logging_path, is_master)
        
    if config['model']['use_pretrain']:
        printlog(f"\t- FLAG: 'use_pretrain' is set 'True'.\n" \
                 f"\t\t* w2v-encoder will be unfrozen from: {w2v_unfrz_step} step" \
                 f" -> (paper: {config['model']['w2v_freeze_steps']} frozen steps / {config['model']['total_steps']} in total)", logging_path, is_master)                    
    else:
        printlog(f"\t- FLAG: 'use_pretrain' is set 'False', The framework will train the model from the scratch.", logging_path, is_master)

    # Calculate epoch approximation
    train_hrs   = config['data']['train_hrs']
    train_size  = train_set.get_total_dataset_size()
    batch_size  = config['common']['batch_size']
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
    FROZEN_WV = True if config['model']['use_pretrain'] else False
    start_eval_no = 1
    last_eval_no  = 3 if is_quickrun else nb_evals
    best_valid_scr = {'EER':9999., 'EER_threshold':1000., 'minDCF':9999., 'DCF_threshold':1000.}
    best_test_scr  = {'EER':9999., 'EER_threshold':1000., 'minDCF':9999., 'DCF_threshold':1000., 
                      'FAR*':9999., 'FRR*':9999., 'EER*':9999.}

    # Zero-shot evaluation if configured
    if config['model']['use_pretrain'] and config['model']['zshot_eval']:
        printlog("FLAG: 'zshot_eval' is set 'True', conducting zero-shot evaluation...", logging_path, is_master)
        if is_ddp: dist.barrier()
        valid_time, valid_scr = evaluate(config, model, valid_loader, device_id)
        test_time,  test_scr  = evaluate(config, model, test_loader,  device_id, threshold=valid_scr['EER_threshold'] if is_master else None)
        printlog('\t---------------------------------------------\n'
                 '\tvalidation time: {:d} min {:02d} sec\n'.format(int(valid_time//60), int(valid_time%60)) +                
                 "\t'valid' | EER:    {:.04f}, threshold: {:.03f}\n".format(valid_scr['EER'], valid_scr['EER_threshold']) +
                 "\t        | minDCF: {:.04f}, threshold: {:.03f}\n".format(valid_scr['minDCF'], valid_scr['DCF_threshold']) +
                 '\t---------------------------------------------\n' +
                 '\ttesting time: {:d} min {:02d} sec\n'.format(int(test_time//60), int(test_time%60)) +
                 "\t'test'  | EER:    {:.04f}, threshold: {:.03f}\n".format(test_scr['EER'], test_scr['EER_threshold']) +
                 "\t        | minDCF: {:.04f}, threshold: {:.03f}\n".format(test_scr['minDCF'], test_scr['DCF_threshold']) +
                 "\t        | FAR*:   {:.04f}, threshold: {:.03f}\n".format(test_scr['FAR'].item(), valid_scr['EER_threshold']) + 
                 "\t        | FRR*:   {:.04f}, -               \n".format(test_scr['FRR'].item()), logging_path, is_master)
        best_valid_scr = {'EER'   :valid_scr['EER'],    'EER_threshold':valid_scr['EER_threshold'],
                          'minDCF':valid_scr['minDCF'], 'DCF_threshold':valid_scr['DCF_threshold']}
        best_test_scr  = {'EER'   : test_scr['EER'],    'EER_threshold': test_scr['EER_threshold'],
                          'minDCF': test_scr['minDCF'], 'DCF_threshold': test_scr['DCF_threshold'],
                          'FAR*': test_scr['FAR'].item(), 'FRR*': test_scr['FRR'].item(), 
                          'EER*': (test_scr['FAR']+test_scr['FRR']).item() / 2.}
        
        if is_neptune and logger is not None and is_master:
            # log the zero-shot scores
            logger['valid/running/EER'].append(valid_scr['EER'], step=0)
            logger['valid/running/EER_threshold'].append(valid_scr['EER_threshold'], step=0)
            logger['valid/running/minDCF'].append(valid_scr['minDCF'], step=0)
            logger['valid/running/DCF_threshold'].append(valid_scr['DCF_threshold'], step=0)
            
            logger['test/running/EER'].append(test_scr['EER'], step=0)
            logger['test/running/threshold'].append(test_scr['EER_threshold'], step=0)
            logger['test/running/minDCF'].append(test_scr['minDCF'], step=0)
            logger['test/running/DCF_threshold'].append(test_scr['DCF_threshold'], step=0)
            
            # track the best scores
            logger['valid/eval/EER'].append(best_valid_scr['EER'], step=0)
            logger['valid/eval/EER_threshold'].append(best_valid_scr['EER_threshold'], step=0)
            logger['valid/eval/minDCF'].append(best_valid_scr['minDCF'], step=0)
            logger['valid/eval/DCF_threshold'].append(best_valid_scr['DCF_threshold'], step=0)
            
            logger['test/eval/EER'].append(best_test_scr['EER'], step=0)
            logger['test/eval/EER_threshold'].append(best_test_scr['EER_threshold'], step=0)
            logger['test/eval/minDCF'].append(best_test_scr['minDCF'], step=0)
            logger['test/eval/DCF_threshold'].append(best_test_scr['DCF_threshold'], step=0)           
            
            logger['test/eval/FAR*'].append(best_test_scr['FAR*'], step=0)
            logger['test/eval/FRR*'].append(best_test_scr['FRR*'], step=0)
            logger['test/eval/EER*'].append(best_test_scr['EER*'], step=0)
        
    # Training iteration starts here
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
            
        if is_ddp: dist.barrier() # Sync the processes before start of the epoch
        FROZEN_WV, train_time, train_scr = train_iteration(FROZEN_WV, w2v_unfrz_step, 
                                                           config, model, train_loader, device_id, optimizer, scheduler)

        # Evaluation Phase
        valid_scr = None
        if not FROZEN_WV or is_quickrun or config['model']['full_eval']:
            valid_time, valid_scr = evaluate(config, model, valid_loader, device_id)
            if is_skiptest:
                test_time, test_scr = 0., best_test_scr
            else:
                test_time, test_scr = evaluate(config, model, test_loader, device_id, threshold=valid_scr['EER_threshold'] if is_master else None)
        
        # Save the best validated model when the best score has been updated
        if not FROZEN_WV and valid_scr is not None and best_valid_scr['EER'] >= valid_scr['EER']: 
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
        
        # print the results out
        printlog('\ttraining time: {:d} min {:02d} sec\n'.format(int(train_time//60), int(train_time%60)) +
                 "\t'train' | ACC: {:.02f}%, F1-macro: {:.02f}%, loss: {:.04f}".format(train_scr['ACC']*100, train_scr['F1']*100, train_scr['loss']), logging_path, is_master)
        if not FROZEN_WV or is_quickrun or config['model']['full_eval']:
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
                        "\t'test'  | EER:    {:.04f}, threshold: {:.03f}".format(test_scr['EER'], test_scr['EER_threshold'])    + " |          EER:    {:.04f}, threshold: {:.03f}\n".format(best_test_scr['EER'], best_test_scr['EER_threshold']) + 
                        "\t        | minDCF: {:.04f}, threshold: {:.03f}".format(test_scr['minDCF'], test_scr['DCF_threshold']) + " |          minDCF: {:.04f}, threshold: {:.03f}\n".format(best_valid_scr['minDCF'], best_valid_scr['DCF_threshold']) +
                        "\t        | FAR*:   {:.04f}, threshold: {:.03f}".format(test_scr['FAR'].item(), valid_scr['EER_threshold'])         + " |          FAR*:   {:.04f}, threshold: {:.03f}\n".format(best_test_scr['FAR*'], best_test_scr['EER_threshold']) + 
                        "\t        | FRR*:   {:.04f},     -           ".format(test_scr['FRR'].item())                          + " |          FRR*:   {:.04f},     -            \n".format(best_test_scr['FRR*']) + 
                        "\t        | EER*:   {:.04f},     -           ".format((test_scr['FAR']+test_scr['FRR']).item() / 2.)   + " |          EER*:   {:.04f},     -            \n".format(best_test_scr['EER*']), logging_path, is_master)
        
        # Make an online log if specified
        if is_neptune and logger is not None and is_master:
            cur_step = scheduler.last_epoch
            
            logger['train/running/loss'].append(train_scr['loss'], step=cur_step)
            logger['train/running/acc'].append(train_scr['ACC'], step=cur_step)
            logger['train/running/f1'].append(train_scr['F1'], step=cur_step)
            
            if not FROZEN_WV:
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
