import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from models.graph_attention import H_Attention_Graph, GCN_normal, GCN_cat
from data.ss_bag import SSBags
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt 
import pandas as pd 


## load model
class inferencer (object):
    def __init__(self, model_path, net, inference_path):
        self.model = net.cuda()
        temp = torch.load(model_path)['net']
        self.model.load_state_dict(temp)
        self.test_transform = transforms.Compose([
                    transforms.Resize(28),
                    transforms.ToTensor()
        ])
        self.testbag = SSBags(inference_path, pre_transform=self.test_transform, sub_list = [5])
        self.batch_size = 1
        self.inference_loader = DataLoader(self.testbag, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test(self):
        self.model.eval()
        for batch_idx, (data, label,idx_list,img_name) in enumerate(tqdm(self.inference_loader, ascii=True, ncols = 60)):   # if label is unavailable, this line shoule be changed
            data = data.cuda()
            label = label.cuda()
            with torch.no_grad():
                prob_label, predicted_label, loss, weights = self.model(data, label, idx_list)
            print(weights)
            print(img_name)
                    
        return weights



if __name__ == '__main__':


    model_path = '/remote-home/gyf/hhy/Ultrasound_MIL/code_results/5folds/Graph/0/ckp/net.ckpt50.pth'
    net = H_Attention_Graph()
    inference_path = '/remote-home/gyf/Ultrasound_CV/data/MergePhase1/5folds'

    A = inferencer(model_path,net,inference_path)
    weights = A.test()
