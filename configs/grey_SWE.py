from __future__ import absolute_import
import os
import sys
sys.path.append('../')

from data.BMDataset import BMDataset
from data.PatientBags import PatientBags
from data.ss_bag import SSBags
from data.bag import BMBags
from data.ruijin import RuijinBags
from models.attentionMIL import Attention, GatedAttention,H_Attention, S_H_Attention
from trainers.MILTrainer import MILTrainer
from utils.logger import Logger
from torch.optim import Adam
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import argparse
from argparse import ArgumentParser
try:
    import collections.abc as collections_abc
except ImportError:
    import collections as collections_abc

def my_collate(batch):
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    label = torch.LongTensor(label)
    return data, label

class Config(object):
    '''
    This config is for single modality BM classification
    '''
    def __init__(self):
        ##The top config
        self.data_root = '/remote-home/my/Ultrasound_CV/data/MergePhase1/test_0.3'
        self.log_dir = '/remote-home/my/hhy/Ultrasound_MIL/code_results/new'

        #self.root = '/remote-home/my/Ultrasound_CV/data/Ruijin/clean'
        #self.log_dir = '/remote-home/my/hhy/Ultrasound_MIL/code_results'
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        ##training config
        self.lr = 1e-4
        self.epoch = 50
        self.resume = -1
        self.batch_size = 1
        self.net = S_H_Attention()
        self.net.cuda()

        self.optimizer = Adam(self.net.parameters(), lr=self.lr)
        self.lrsch = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 30, 50, 70], gamma=0.5)

        
        self.logger = Logger(self.log_dir)

        self.train_transform = transforms.Compose([
                    transforms.Resize(224),
                    #transforms.ColorJitter(brightness = 0.25),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    # transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
                    transforms.ToTensor()
        ])
        self.test_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor()
        ])

        self.trainbag = SSBags(self.data_root+'/train',pre_transform=self.train_transform)
        self.testbag = SSBags(self.data_root+'/test', pre_transform=self.test_transform)

        # patient bags
        #self.trainbag = PatientBags(self.data_root+'/train', self.train_transform)
        #self.testbag = PatientBags(self.data_root+'/test', self.test_transform)
        '''
        # random 
        
        self.train_set = BMDataset(self.data_root+'/train',self.train_transform)
        self.test_set = BMDataset(self.data_root+'/test',self.test_transform)
        self.trainbag = BMBags(dataset=self.train_set)
        self.testbag = BMBags(dataset=self.test_set)
        '''

        #self.train_loader = DataLoader(self.trainbag, batch_size=1, shuffle=True, num_workers=8)
        #self.val_loader = DataLoader(self.testbag, batch_size=1, shuffle=False, num_workers=8)
        self.train_loader = DataLoader(self.trainbag, batch_size=1, shuffle=True, num_workers=8)
        #print (len(self.train_loader))
        self.val_loader = DataLoader(self.testbag, batch_size=1, shuffle=False, num_workers=8)

        if self.resume > 0:
            self.net, self.optimizer, self.lrsch, self.loss, self.global_step = self.logger.load(self.net, self.optimizer, self.lrsch, self.loss, self.resume)
        else:
            self.global_step = 0
        
        # self.trainer = MTTrainer(self.net, self.optimizer, self.lrsch, self.loss, self.train_loader, self.val_loader, self.logger, self.global_step, mode=2)
        self.trainer = MILTrainer(self.net, self.optimizer, self.lrsch, None, self.train_loader, self.val_loader, self.logger,
                                     self.global_step)
    def update(self,dicts):
        for k,v in dicts.items():
            self.__setattr__(k,v)

    @staticmethod
    def auto_argparser(cfg,description=None):
        """Generate argparser from config file automatically (experimental)
        """
        # partial_parser = ArgumentParser(description=description)
        # partial_parser.add_argument('config', help='config file path')
        # parsed, unparsed_args = partial_parser.parse_known_args()
        # cfg = Config.fromfile(parsed.config)
        parser = ArgumentParser(description=description)
        add_args(parser, cfg)
        parsed_args = parser.parse_args()
        update_cfg(cfg, parsed_args)
        return cfg

        

# config = Config().__dict__

def add_args(parser, cfg, prefix=''):
    '''
    cfg: dict
    '''
    for k, v in cfg.items():
        if v is None or isinstance(v, str):
            parser.add_argument('--' + prefix + k,type=str,default=v)
        elif isinstance(v, bool):
            parser.add_argument('--' + prefix + k, type=str, default=v,
                choices=('True', 'False', 'true', 'false'))
        elif isinstance(v, int):
            parser.add_argument('--' + prefix + k, type=int,default=v)
        elif isinstance(v, float):
            parser.add_argument('--' + prefix + k, type=float,default=v)
        elif isinstance(v, dict):
            add_args(parser, v, prefix + k + '.')
        elif isinstance(v, collections_abc.Iterable):
            parser.add_argument('--' + prefix + k, type=type(v[0]), nargs='+',default=v)
        else:
            print('connot parse key {} of type {}'.format(prefix + k, type(v)))
    return parser

def update_cfg(cfg, parsed_args):

    def convert(new_cfgdict, parsed_args):
        for k, v in parsed_args.items():
            topk = k.split('.')
            if len(topk) >= 2:
                convert(new_cfgdict[topk[0]], {'.'.join(topk[1:]): v})
            elif v is not None:
                if v in ['True', 'False', 'true', 'false']:
                    if v.title() == 'True': v = True
                    else: v = False
                new_cfgdict[topk[0]] = v

    new_cfgdict = Dict()
    convert(new_cfgdict, parsed_args.__dict__)
    cfg.update(new_cfgdict.to_dict())



if __name__=='__main__':
    config = Config()
    print(config.__dict__)
    
    
