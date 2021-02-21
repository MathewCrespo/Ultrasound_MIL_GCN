import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from models.attentionMIL import Attention, GatedAttention, MIL
from data.PatientBags import PatientBags
from data.bag import BMBags
from data.BMDataset import BMDataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt 
import pandas as pd 


## load model
class inferencer (object):
    def __init__(self, model_path, net, inference_path, pre_transform):
        self.model = net.cuda()
        temp = torch.load(model_path)['net']
        self.model.load_state_dict(temp)
        self.inference_bag = PatientBags(inference_path, pre_transform,mix_mode=1)
        '''
        inference_set = BMDataset(inference_path,pre_transform)
        self.inference_bag = BMBags(inference_set)
        '''
        self.batch_size = 1
        self.inference_loader = DataLoader(self.inference_bag, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test(self):
        self.model.eval()
        pred = []
        prob = []
        g_truth = []
        total_weights = []
        replace_record = []
        for batch_idx, (data, label,replace_num) in enumerate(tqdm(self.inference_loader, ascii=True, ncols = 60)):   # if label is unavailable, this line shoule be changed
            data = data.cuda()
            replace_record.append(replace_num.item())
            g_truth.append(label)
            with torch.no_grad():
                prob_label, predicted_label, attention_weights = self.model.calculate_weights(data)
            total_weights.append(attention_weights)
            # label or bag label?
            #target.append(bag_label.cpu().detach().numpy().ravel())
            pred.append(predicted_label.cpu().detach().numpy().ravel())
            prob.append(prob_label.cpu().detach().numpy().ravel())
        return replace_record, g_truth, prob, pred, total_weights

    def inference(self):
        self.replace_record, self.g_truth, self.prob_label, self.pred_label, self.total_weights = self.test()

    def stats (self):
        pos_bags = []
        neg_bags = []
        all_pos_weights = []
        for i in range(len(self.pred_label)):
            # neg bag 
            if self.g_truth[i].item() == 0:
                neg_bags.append(self.prob_label[i].item())
                
            # pos bag
            else:
                pos_bags.append(self.prob_label[i].item())  # stats on bag scores
                replace_num = self.replace_record[i]
                weights = self.total_weights[i].squeeze(-2).tolist()
                l = len(weights)
                pos_weights = weights[0:l-replace_num]
                #print(replace_num)
                all_pos_weights.extend(pos_weights)

        '''
        df = pd.DataFrame({'x1':pos_bags})
        df.plot(kind='hist')
        df.plot(kind = 'kde')
        plt.show()
        '''
        plt.hist(all_pos_weights,range=[0,1])
        plt.show()
    
        '''
        plt.hist(pos_bags, range=[0,1],density=False)
        plt.show()
        plt.hist(neg_bags,range=[0,1], color='g',density=False)
        plt.show()
        '''

        return pos_bags, neg_bags





if __name__ == '__main__':


    model_path = '/media/hhy/data/code_results/MILs/MIL_MixedBags/ckp/net.ckpt50.pth'
    net = Attention()
    inference_path = '/media/hhy/data/USdata/MergePhase1/test_0.3/test'
    pre_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor()
        ])
    A = inferencer(model_path,net,inference_path, pre_transform)
    A.inference()
    A.stats()
    '''
    for i in range (len(A.total_weights)):
        print('Predicted label is {}'.format(A.pred_label[i].item()))
        print('Weights are {}'.format(A.total_weights[i].squeeze(-2).tolist()))
    #print(A.total_weights)
    '''