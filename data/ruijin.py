from PIL import Image
import torch
import os
import sys
from torch.utils.data import Dataset
from torchvision import transforms
import xml.etree.cElementTree as ET
from tqdm import tqdm
import random
import pandas as pd
from random import randint,sample


class RuijinBags(Dataset):
    def __init__(self, root, sub_list, pre_transform, crop_mode=False, mix_mode=0,
                 label_name="手术淋巴结情况（0未转移；1转移）", clinical_list=None, info_type = 1):
        self.root = root
        self.label_name = label_name
        self.clinical_list = clinical_list
        self.sub_list = sub_list
        self.pre_transform = pre_transform
        self.patient_info = [] # each patient is a dict {'ID':ID,'label','image_path':, 'fold':}
        self.table = pd.read_excel('/remote-home/my/Ultrasound_CV/data/Ruijin/phase1/data.xlsx', keep_default_na=False)
        self.info_type =  info_type  # 0 -- images only  1-- clinical info only  2-- combined
        # create patient_info based on 5 fold
        for fold in self.sub_list:
            self.scan(fold)

    def scan(self, fold):  # to create entire patient info (list)
        fold_table = self.table[self.table['fold'] == fold].reset_index(drop=True)
        for k in range(len(fold_table)):
            ID = fold_table.loc[k,'ID']
            #print(ID)
            img_path = os.path.join(self.root, ID)
            imgs = os.listdir(img_path)
            label = fold_table.loc[k, self.label_name]
            if self.clinical_list != None:
                clinical_info = []
                for info in self.clinical_list:
                    if fold_table.loc[k, info] == '':
                        clinical_info.append(0)
                    else:
                        clinical_info.append(fold_table.loc[k, info])
            else:
                clinical_info = None
            label = label -1 if self.label_name== "pLN分组3（1为0-2枚淋巴结转移；2为＞2枚淋巴结转移）" else label
            now_patient = {
                'ID': ID,
                'imgs': [img_path + '/'+ img for img in imgs],
                'label': label,
                'clinical_info': clinical_info
            }
            self.patient_info.append(now_patient)
            
    def __getitem__(self, idx):
        now_patient = self.patient_info[idx]
        label = now_patient['label']
        imgs = []
        for img_path in now_patient['imgs']:
            img = Image.open(img_path).convert('RGB')
            if self.pre_transform is not None:
                img = self.pre_transform(img)
            imgs.append(img)

        if self.clinical_list is not None:
            clinical_info = torch.Tensor(now_patient['clinical_info'])

        if self.info_type == 2:  # combined    
            return torch.stack([x for x in imgs], dim=0), clinical_info, label

        elif self.info_type == 1:  # clinical_info only
            return clinical_info, label

        else: # images only
            return torch.stack([x for x in imgs], dim=0), label
    def __len__(self):
        return len(self.patient_info)


if __name__ == '__main__':
    root = '/remote-home/my/Ultrasound_CV/data/Ruijin/clean'
    pre_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    ])
    clinical_list = ["T-size.1", "多灶性（1是；0否）", "T-stage（T1≤2cm；T2＞2cm）",
                          "病理分类:0非ILC；1ILC", "病理分级：3=III级；2=II级；1=I级；0=unkown级",
                          "淋巴管癌栓：0No；1Yes", "ER（0阴性；1阳性）", "PR ≥20%（0阴性；1阳性）",
                          "ER/PR(0为全阴；1）", "ki67：1≥14%；0＜14%", "HER-2(0阴性；1阳性）",
                          "免疫分型（1A；2B；3:Her-2 过表达；4:三阴性）", "年龄",
                          "异常淋巴结：1有0无", "异常淋巴结：1-I区；2-腋窝其他区；3累及内乳及同侧锁骨上",
                          "可疑淋巴结数目：0,1，2，3为＞2", "皮质厚度（mm）", "淋巴门：1有；0无",
                          "可疑程度（0无可疑；1轻度可疑；2中度可疑；3重度可疑）"]
    all_set = RuijinBags(root, [0,1,2,3,4],pre_transform, clinical_list=clinical_list,info_type=1)

    print(all_set[0][0].shape)
    '''
    label_set = []
    for i in range(len(all_set)):
        label = all_set[i][1]
        label_set.append(label)
    #print(label_set.count(1))
    #print(label_set)
    '''

