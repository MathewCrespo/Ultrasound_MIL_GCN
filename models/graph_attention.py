import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGPooling
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp



class FE(nn.Module):
    def __init__(self, input_dim=3, L=500, D=128, K=1):
        super(FE, self).__init__()
        self.input_dim = input_dim
        self.L = L
        self.D = D
        self.K = K

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)       
        H = H.view(-1, 50 * 4* 4)       
        H = self.feature_extractor_part2(H)  # NxL
        return H

class GCN_H (nn.Module):  
    def __init__(self,num_features=500, nhid=256, num_classes=2, pooling_ratio = 0.75):
        super(GCN_H,self).__init__()
        self.num_features = num_features
        self.nhid = nhid
        self.num_classes = num_classes
        self.pooling_ratio = pooling_ratio
        
        self.conv1 = GCNConv(500, self.nhid)
        #print(1)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio)

    def get_threshold (self,x):

        gamma = 0
        '''
        node_num = x.shape[0]
        for i in range(node_num):
            f_dist = torch.sum((x-x[i,:])**2,dim=1)
            temp_max,_ = torch.max(f_dist,dim=0)
            gamma = max(gamma,temp_max.item())
        '''
        return gamma

    def get_edge_index(self,x):        
        t = self.get_threshold(x)
        node_num = x.shape[0]
        edge_index = [[],[]] # source nodes and target nodes
        for i in range(node_num):
            f_dist = torch.sum((x-x[i,:])**2,dim=1)
            index = (f_dist < t)
            #print(index)
            for j in range(i+1,node_num):
                if index[j]:

                    edge_index[0].append(i) #source
                    edge_index[1].append(j)

        return torch.LongTensor(edge_index).cuda()
    
    def forward(self, feature):
        edge_index = self.get_edge_index(feature)        
        x = F.relu(self.conv1(feature,edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch,_, _ = self.pool2(x, edge_index)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3   # image feature after SAGPooling

        return x

class Attention(nn.Module):
    def __init__(self, D=128, K=1):
        super(Attention, self).__init__()
        self.D = D
        self.K = K

        self.attention_layer = nn.Sequential(
            nn.Linear(512, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
        nn.Linear(512, 1),
        nn.Sigmoid()
        )

    def forward(self, x):
        A = self.attention_layer(x)
        A = torch.transpose(A, 0, 1)
        A = F.softmax(A, dim=1)

        bag_f = torch.mm(A, x)

        prob = self.classifier(bag_f)
        pred = torch.ge(prob, 0.5).float()

        return prob, pred, A
    
    def cal_loss(self, X, Y):
        Y = Y.float()
        Y_prob, Y_pred, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))

        return Y_prob, Y_pred, neg_log_likelihood, A

class GCN_normal (nn.Module):  
    def __init__(self,num_features=500, nhid=256, pooling='mean'):

        super(GCN_normal,self).__init__()
        self.num_features = num_features
        self.nhid = nhid
        self.pooling = pooling
        self.conv1 = GCNConv(500, self.nhid)
        #print(1)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.conv3 = GCNConv(self.nhid, self.nhid)


    def get_threshold (self,x):
        gamma = 0
        node_num = x.shape[0]
        for i in range(node_num):
            f_dist = torch.sum((x-x[i,:])**2,dim=1)
            temp_max,_ = torch.max(f_dist,dim=0)
            gamma = max(gamma,temp_max.item())
            #print(1)
        return gamma*0.5

    def get_edge_index(self,x):        
        t = self.get_threshold(x)
        node_num = x.shape[0]
        edge_index = [[],[]] # source nodes and target nodes
        for i in range(node_num):
            f_dist = torch.sum((x-x[i,:])**2,dim=1)
            index = (f_dist < t)
            #print(index)
            for j in range(i+1,node_num):
                if index[j]:

                    edge_index[0].append(i) #source
                    edge_index[1].append(j)

        return torch.LongTensor(edge_index).cuda()
    
    def forward(self, feature):
        edge_index = self.get_edge_index(feature)        
        x = F.relu(self.conv1(feature,edge_index))
        
        if self.pooling == 'mean':
            x1 = torch.mean(x,dim=0).unsqueeze(0)
            x1 = torch.cat([x1,x1],dim=1)
            #print(x1.shape)
        
        if self.pooling == 'max':
            x1, _ = torch.max(x,dim=0)
            x1 = x1.unsqueeze(0)
            x1 = torch.cat([x1,x1],dim=1)


        x = F.relu(self.conv2(x, edge_index))

        if self.pooling == 'mean':
            x2 = torch.mean(x,dim=0).unsqueeze(0)
            x2 = torch.cat([x2,x2],dim=1)

        if self.pooling == 'max':
            x2, _ = torch.max(x,dim=0)
            x2 = x2.unsqueeze(0)
            x2 = torch.cat([x2,x2],dim=1)
        
        x = F.relu(self.conv3(x, edge_index))
       
        if self.pooling == 'mean':
            x3 = torch.mean(x,dim=0).unsqueeze(0)
            x3 = torch.cat([x3,x3],dim=1)
        
        if self.pooling == 'max':
            x3, _ = torch.max(x,dim=0)
            x3 = x3.unsqueeze(0)
            x3 = torch.cat([x3,x3],dim=1)
            
        x = x1 + x2 + x3   # image feature after SAGPooling

        return x


class GCN_cat (nn.Module):  
    def __init__(self,num_features=500, nhid=256):

        super(GCN_cat,self).__init__()
        self.num_features = num_features
        self.nhid = nhid
        self.conv1 = GCNConv(500, self.nhid)
        #print(1)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.conv3 = GCNConv(self.nhid, self.nhid)


    def get_threshold (self,x):
        gamma = 0
        node_num = x.shape[0]
        for i in range(node_num):
            f_dist = torch.sum((x-x[i,:])**2,dim=1)
            temp_max,_ = torch.max(f_dist,dim=0)
            gamma = max(gamma,temp_max.item())
            #print(1)
        return gamma*0.5

    def get_edge_index(self,x):        
        t = self.get_threshold(x)
        node_num = x.shape[0]
        edge_index = [[],[]] # source nodes and target nodes
        for i in range(node_num):
            f_dist = torch.sum((x-x[i,:])**2,dim=1)
            index = (f_dist < t)
            #print(index)
            for j in range(i+1,node_num):
                if index[j]:

                    edge_index[0].append(i) #source
                    edge_index[1].append(j)

        return torch.LongTensor(edge_index).cuda()
    
    def forward(self, feature):
        edge_index = self.get_edge_index(feature)        
        x = F.relu(self.conv1(feature,edge_index))
        
        x1_mean = torch.mean(x,dim=0).unsqueeze(0)
        x1_max,_ = torch.max(x,dim=0)
        x1_max = x1_max.unsqueeze(0)
        x1 = torch.cat([x1_max,x1_mean],dim=1)

        x = F.relu(self.conv2(x, edge_index))

        x2_mean = torch.mean(x,dim=0).unsqueeze(0)
        x2_max,_ = torch.max(x,dim=0)
        x2_max = x2_max.unsqueeze(0)
        x2 = torch.cat([x2_max,x2_mean],dim=1)
        
        x = F.relu(self.conv3(x, edge_index))
       
        x3_mean = torch.mean(x,dim=0).unsqueeze(0)
        x3_max,_ = torch.max(x,dim=0)
        x3_max = x3_max.unsqueeze(0)
        x3 = torch.cat([x3_max,x3_mean],dim=1)
        
            
        x = x1 + x2 + x3   # image feature after SAGPooling

        return x

class H_Attention_Graph(nn.Module):
    def __init__(self, fe = FE() , gcn = GCN_H(), attn = Attention()):
        super(H_Attention_Graph,self).__init__()
        self.fe = fe
        self.gcn = gcn
        self.attn = attn
        
    def forward(self, x, y, idx_list):
        #print('Input Shape is ', x.shape)
        # part1: feature extractor
        H = self.fe(x)
        #print('Patch Feature Shape is ', H.shape)

        # part2: construct a graph and get image level feature
        img_features = []
        for i in range(len(idx_list)-1):
            h_patch = H[idx_list[i]:idx_list[i+1], :]
            i_feature = self.gcn(h_patch.cuda())
            #print(i_feature.shape)
            img_features.append(i_feature)

        instance_f = torch.cat([x for x in img_features],dim = 0)
        #print('Image Feature Shape is ', instance_f.shape)

        # part3 bag aggregation and prediction
        Y_prob, Y_pred, loss, weights = self.attn.cal_loss(instance_f,y)

        return Y_prob, Y_pred, loss, weights





            
            
            

        