import random
import numpy as np
import os
import json
import argparse
import torch
import dataloaders
# from models.model_reco import VCL
from models.model_vid import VCL
import math
import copy
from utils import Logger
from trainer_vid import TrainerVid
from trainer import Trainer
import torch.nn.functional as F
from utils.losses import abCE_loss, CE_loss, consistency_weight

import torch.multiprocessing as mp
import torch.distributed as dist


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(gpu, ngpus_per_node, config, resume, test):
    if gpu == 0:
        train_logger = Logger()
        test_logger = Logger()
    else:
        train_logger = None
        test_logger = None

    config['rank'] = gpu + ngpus_per_node * config['n_node']
    # iter_per_epoch = 86
    iter_per_epoch = config['n_labeled_examples'] * config['n_unlabeled_ratio'] // config['train_unsupervised'][
        'batch_size']
    torch.cuda.set_device(gpu)
    assert config['train_supervised']['batch_size'] % config['n_gpu'] == 0
    assert config['train_unsupervised']['batch_size'] % config['n_gpu'] == 0
    assert config['train_vid']['batch_size'] % config['n_gpu'] == 0
    assert config['val_loader']['batch_size'] % config['n_gpu'] == 0
    config['train_supervised']['batch_size'] = int(config['train_supervised']['batch_size'] / config['n_gpu'])
    config['train_unsupervised']['batch_size'] = int(config['train_unsupervised']['batch_size'] / config['n_gpu'])
    config['train_vid']['batch_size'] = int(config['train_vid']['batch_size'] / config['n_gpu'])
    config['val_loader']['batch_size'] = int(config['val_loader']['batch_size'] / config['n_gpu'])
    config['train_supervised']['num_workers'] = int(config['train_supervised']['num_workers'] / config['n_gpu'])
    config['train_unsupervised']['num_workers'] = int(config['train_unsupervised']['num_workers'] / config['n_gpu'])
    config['train_vid']['num_workers'] = int(config['train_vid']['num_workers'] / config['n_gpu'])
    config['val_loader']['num_workers'] = int(config['val_loader']['num_workers'] / config['n_gpu'])
    dist.init_process_group(backend='nccl', init_method=config['dist_url'], world_size=config['world_size'],
                            rank=config['rank'])

    seed = config['random_seed']

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # DATA LOADERS
    config['train_supervised']['n_labeled_examples'] = config['n_labeled_examples']
    config['train_supervised']['n_unlabeled_ratio'] = config['n_unlabeled_ratio']
    config['train_vid']['n_labeled_examples'] = config['n_labeled_examples']
    config['train_vid']['n_unlabeled_ratio'] = config['n_unlabeled_ratio']
    config['train_vid']['clip_size'] = config['clip_size']
    config['train_unsupervised']['n_labeled_examples'] = config['n_labeled_examples']
    config['train_unsupervised']['n_unlabeled_ratio'] = config['n_unlabeled_ratio']
    config['train_unsupervised']['use_weak_lables'] = config['use_weak_lables']
    config['train_supervised']['data_dir'] = config['data_dir']
    config['train_unsupervised']['data_dir'] = config['data_dir']
    config['train_vid']['data_dir'] = config['data_dir']
    config['val_loader']['data_dir'] = config['data_dir']
    config['train_supervised']['datalist'] = config['datalist']
    config['train_unsupervised']['datalist'] = config['datalist']
    config['train_vid']['datalist'] = config['datalist']
    config['val_loader']['datalist'] = config['datalist']
    config['test_loader'] = copy.deepcopy(config['val_loader'])
    config['test_loader']['split'] = 'test'
    config['test_loader']['num_workers'] = 1
    if config['dataset'] == 'voc':
        sup_dataloader = dataloaders.VOC
        unsup_dataloader = dataloaders.PairVOC
    elif config['dataset'] == 'cityscapes':
        sup_dataloader = dataloaders.City
        unsup_dataloader = dataloaders.PairCity
    elif config['dataset'] == 'thermal':
        sup_dataloader = dataloaders.Thermal
        unsup_dataloader = dataloaders.PairThermal
    elif config['dataset'] == 'thermalseq':
        sup_dataloader = dataloaders.ThermalSeq
        unsup_dataloader = dataloaders.PairThermalSeq
    elif config['dataset'] == 'thermalour':
        sup_dataloader = dataloaders.ThermalOur
        unsup_dataloader = dataloaders.PairThermalOur
    elif config['dataset'] == 'thermalvid':
        sup_dataloader = dataloaders.ThermalVid
        unsup_dataloader = dataloaders.PairThermalVid
        vid_loader = dataloaders.ClipThermalVid

    supervised_loader = sup_dataloader(config['train_supervised'])
    unsupervised_loader = unsup_dataloader(config['train_unsupervised'])
    clip_loader = vid_loader(config['train_vid'])
    val_loader = sup_dataloader(config['val_loader'])
    test_loader = sup_dataloader(config['test_loader'])

    sup_loss = CE_loss
    model = VCL(num_classes=val_loader.dataset.num_classes, conf=config['model'],
                       sup_loss=sup_loss, ignore_index=val_loader.dataset.ignore_index)
    if gpu == 0:
        print(f'\n{model}\n')

    # TRAINING
    trainer = TrainerVid(
        model=model,
        resume=resume,
        config=config,
        supervised_loader=supervised_loader,
        unsupervised_loader=unsupervised_loader,
        clip_loader=clip_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        iter_per_epoch=iter_per_epoch,
        train_logger=train_logger,
        test_logger=test_logger,
        gpu=gpu,
        test=test)

    trainer.train()


def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.

    return port


if __name__ == '__main__':
    # PARSE THE ARGS
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/thermalseq_cac_deeplabv3+_resnet101_1over8_datalist0.json',type=str,
                        help='Path to the config file')
    # parser.add_argument('-r', '--resume', default='runs/thermalvid_cac_deeplabv3+_resnet50_1over4_datalist0/04-20_15-56/best_model.pth', type=str,
    #                     help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-r', '--resume', default='', type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-t', '--test', default=False, type=bool,
                        help='whether to test')
    args = parser.parse_args()

    config = json.load(open(args.config))
    torch.backends.cudnn.benchmark = True
    # port = find_free_port()
    port = '52234'
    config['dist_url'] = f"tcp://127.0.0.1:{port}"
    config['n_node'] = 0  # only support 1 node
    config['world_size'] = config['n_gpu']
    mp.spawn(main, nprocs=config['n_gpu'], args=(config['n_gpu'], config, args.resume, args.test))



