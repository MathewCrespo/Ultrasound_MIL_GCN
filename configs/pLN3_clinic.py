from __future__ import absolute_import
import os
import sys

sys.path.append('../')

from data.BMDataset import BMDataset
from data.PatientBags import PatientBags
from data.bag import BMBags
from data.ruijin import RuijinBags
from models.attentionMIL import Attention, GatedAttention, MIL, Res_Attention, H_Attention, C_Attention
from trainers.MILCTrainer import MILCTrainer
from utils.logger import Logger
from torch.optim import Adam
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler
import torch
import torch.nn as nn
import argparse
from argparse import ArgumentParser

try:
    import collections.abc as collections_abc
except ImportError:
    import collections as collections_abc


class Config(object):
    '''
    This config is for single modality BM classification
    '''

    def __init__(self):
        ##The top config
        # self.data_root = '/media/hhy/data/USdata/MergePhase1/test_0.3'
        # self.log_dir = '/media/hhy/data/code_results/MILs/MIL_H_Attention'

        self.root = '/remote-home/my/Ultrasound_CV/data/Ruijin/clean'
        self.log_dir = '/remote-home/my/lys/MIL/experiments/weighted_sampler_pLN3/'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.clinical_list = ["T-size.1", "多灶性（1是；0否）", "T-stage（T1≤2cm；T2＞2cm）",
                              "病理分类:0非ILC；1ILC", "病理分级：3=III级；2=II级；1=I级；0=unkown级",
                              "淋巴管癌栓：0No；1Yes", "ER（0阴性；1阳性）", "PR ≥20%（0阴性；1阳性）",
                              "ER/PR(0为全阴；1）", "ki67：1≥14%；0＜14%", "HER-2(0阴性；1阳性）",
                              "免疫分型（1A；2B；3:Her-2 过表达；4:三阴性）", "年龄",
                              "异常淋巴结：1有0无", "异常淋巴结：1-I区；2-腋窝其他区；3累及内乳及同侧锁骨上",
                              "可疑淋巴结数目：0,1，2，3为＞2", "皮质厚度（mm）", "淋巴门：1有；0无",
                              "可疑程度（0无可疑；1轻度可疑；2中度可疑；3重度可疑）"]
        self.clinical_len = len(self.clinical_list)
        ##training config
        self.lr = 1e-4
        self.epoch = 50
        self.resume = -1
        self.batch_size = 1
        self.net = C_Attention(C=self.clinical_len)
        self.net.cuda()

        self.optimizer = Adam(self.net.parameters(), lr=self.lr)
        self.lrsch = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 30, 50, 70], gamma=0.5)

        self.logger = Logger(self.log_dir)
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
            transforms.ToTensor()
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.label_name = "pLN分组3（1为0-2枚淋巴结转移；2为＞2枚淋巴结转移）"

        self.trainbag = RuijinBags(self.root, [0, 1, 2, 3], self.train_transform,
                                   label_name=self.label_name, clinical_list=self.clinical_list)
        self.testbag = RuijinBags(self.root, [4], self.test_transform,
                                  label_name=self.label_name, clinical_list=self.clinical_list)

        train_label_list = list(map(lambda x: int(x['label']), self.trainbag.patient_info))
        pos_ratio = sum(train_label_list) / len(train_label_list)
        print(pos_ratio)
        train_weight = [(1 - pos_ratio) if x > 0 else pos_ratio for x in train_label_list]

        self.train_sampler = WeightedRandomSampler(weights=train_weight, num_samples=len(self.trainbag))
        self.train_loader = DataLoader(self.trainbag, batch_size=self.batch_size, num_workers=8,
                                       sampler=self.train_sampler)
        self.val_loader = DataLoader(self.testbag, batch_size=self.batch_size, shuffle=False, num_workers=8)

        if self.resume > 0:
            self.net, self.optimizer, self.lrsch, self.loss, self.global_step = self.logger.load(self.net,
                                                                                                 self.optimizer,
                                                                                                 self.lrsch, self.loss,
                                                                                                 self.resume)
        else:
            self.global_step = 0

        # self.trainer = MTTrainer(self.net, self.optimizer, self.lrsch, self.loss, self.train_loader, self.val_loader, self.logger, self.global_step, mode=2)
        self.trainer = MILCTrainer(self.net, self.optimizer, self.lrsch, None, self.train_loader, self.val_loader,
                                  self.logger,
                                  self.global_step)

    def update(self, dicts):
        for k, v in dicts.items():
            self.__setattr__(k, v)

    @staticmethod
    def auto_argparser(cfg, description=None):
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
            parser.add_argument('--' + prefix + k, type=str, default=v)
        elif isinstance(v, bool):
            parser.add_argument('--' + prefix + k, type=str, default=v,
                                choices=('True', 'False', 'true', 'false'))
        elif isinstance(v, int):
            parser.add_argument('--' + prefix + k, type=int, default=v)
        elif isinstance(v, float):
            parser.add_argument('--' + prefix + k, type=float, default=v)
        elif isinstance(v, dict):
            add_args(parser, v, prefix + k + '.')
        elif isinstance(v, collections_abc.Iterable):
            parser.add_argument('--' + prefix + k, type=type(v[0]), nargs='+', default=v)
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
                    if v.title() == 'True':
                        v = True
                    else:
                        v = False
                new_cfgdict[topk[0]] = v

    new_cfgdict = Dict()
    convert(new_cfgdict, parsed_args.__dict__)
    cfg.update(new_cfgdict.to_dict())


if __name__ == '__main__':
    config = Config()
    print(config.__dict__)


