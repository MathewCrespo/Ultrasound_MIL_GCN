import torch
import torch.nn as nn
import torch.nn.functional as F
from .ResNet import ResNet10, ResNet18

class MIL(nn.Module):    # normal MIL without attention 
    # L - feature dimensions D - variance in attention layer
    def __init__(self, input_dim=3, L=500, D=128, K=1):
        super(MIL, self).__init__()
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
            #nn.AdaptiveMaxPool2d(4)
        )
        # L-dimension feature for each instance
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 53 * 53, self.L),
            nn.ReLU(),
        )
        ### change here
        '''
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        ''' 
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)

        H = H.view(-1, 50 * 53 * 53)

        H = self.feature_extractor_part2(H)     # NxL

        '''
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        ## A -- weights

        M = torch.mm(A, H)  # KxL
        '''
        #print(H.shape)

        

        # no attention -- average pooling   Nx1
        Y_prob = self.classifier(H)
        
    
        Y_prob,_= torch.max(Y_prob, dim=0)
        #print("Y_prob:",Y_prob)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, H

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _,M = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, Y_prob, M

## Hierarchical Attention Multiple Instance Learning

class H_Attention(nn.Module):
    def __init__(self, input_dim=3, L=500, D=128, K=1):
        super(H_Attention, self).__init__()
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
            ##nn.AdaptiveMaxPool2d(4)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x, debug=False):
        if debug:
            print("input shape:", x.shape)
        x = x.squeeze(0)
        x = x.view(-1,3,28,28)
        if debug:
            print("squeeze shape:", x.shape)

        H = self.feature_extractor_part1(x)
        if debug:
            print("feature_extractor_part1 shape:", H.shape)
        H = H.view(-1, 50 * 4 * 4)
        if debug:
            print("view shape:", H.shape)
        H = self.feature_extractor_part2(H)  # NxL
        if debug:
            print("feature_extractor_part2 shape:", H.shape)

        H = H.view(-1,64,500)   # N x P x L
        if debug:
            print ('feature permutation:',H.shape)

        ## doing attention on image patch level

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 2)  # KxN
        A = F.softmax(A, dim=2)  # softmax over N
        if debug:
            print('weights shape:', A.shape)
            check = A.squeeze(1)
            total = torch.sum(check,dim=1)
            print('check patch weights:',total)
        ins_feature = torch.bmm(A, H)  # KxL          NxL  aggregated features of each image
          # Nx1xL

        ins_feature = ins_feature.squeeze(1)
        if debug:
            print('image feature:', ins_feature.shape)
        ## doing attention on bag level
        B_attention = self.attention(ins_feature)
        B_attention = torch.transpose(B_attention, 0, 1)
        B_attention = F.softmax(B_attention, dim=1)
        if debug:
            print('b attention:', B_attention.shape)
        bag_feature = torch.mm(B_attention, ins_feature)
        if debug:
            print('bag feature:', bag_feature.shape)

        Y_prob = self.classifier(bag_feature)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        #print("Y_prob is：", Y_prob)
        return neg_log_likelihood, Y_prob, A
    
    def calculate_weights(self, X):
        Y_prob, Y_hat, weights = self.forward(X)
        Y_prob  = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        return Y_prob, Y_hat, weights



### attention multiple learning
class Attention(nn.Module):
    def __init__(self, input_dim=3, L=500, D=128, K=1):
        super(Attention, self).__init__()
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
            ##nn.AdaptiveMaxPool2d(4)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 53 * 53, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x, debug=False):
        if debug:
            print("input shape:", x.shape)
        x = x.squeeze(0)
        if debug:
            print("squeeze shape:", x.shape)

        H = self.feature_extractor_part1(x)
        if debug:
            print("feature_extractor_part1 shape:", H.shape)
        H = H.view(-1, 50 * 53 * 53)
        if debug:
            print("view shape:", H.shape)
        H = self.feature_extractor_part2(H)  # NxL
        if debug:
            print("feature_extractor_part2 shape:", H.shape)

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        #print("Y_prob is：", Y_prob)
        return neg_log_likelihood, Y_prob, A
    
    def calculate_weights(self, X):
        Y_prob, Y_hat, weights = self.forward(X)
        Y_prob  = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        return Y_prob, Y_hat, weights

class GatedAttention(nn.Module):
    def __init__(self,input_dim=3, L=500, D=128, K=1):
        super(GatedAttention, self).__init__()
        self.input_dim = input_dim
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
            #nn.AdaptiveMaxPool2d(4)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 53 * 53, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 53 * 53)
        H = self.feature_extractor_part2(H)  # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, Y_prob, A

class Res_Attention(nn.Module):
    def __init__(self, input_dim=3, L=500, D=128, K=1):
        super(Res_Attention, self).__init__()
        self.input_dim = input_dim
        self.L = L
        self.D = D
        self.K = K

        self.feature_extractor_part1 = ResNet18()

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(512 * 7 * 7, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x, debug=False):
        if debug:
            print("input shape:", x.shape)
        x = x.squeeze(0)
        if debug:
            print("squeeze shape:", x.shape)

        H = self.feature_extractor_part1(x)
        if debug:
            print("feature_extractor_part1 shape:", H.shape)
        H = H.view(-1, 512 * 7 * 7)
        if debug:
            print("view shape:", H.shape)
        H = self.feature_extractor_part2(H)  # NxL
        if debug:
            print("feature_extractor_part2 shape:", H.shape)

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        #print("Y_prob is：", Y_prob)
        return neg_log_likelihood, Y_prob, A


class C_Attention(nn.Module):
    def __init__(self, input_dim=3, L=500, D=128, K=1, C=0):
        super(C_Attention, self).__init__()
        self.input_dim = input_dim
        self.L = L
        self.D = D
        self.K = K
        self.C = C

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
            ##nn.AdaptiveMaxPool2d(4)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 53 * 53, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L + self.C, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear((self.L+self.C) * self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x, c, debug=False):
        if debug:
            print("input shape:", x.shape)
        x = x.squeeze(0)
        if debug:
            print("squeeze shape:", x.shape)

        H = self.feature_extractor_part1(x)
        if debug:
            print("feature_extractor_part1 shape:", H.shape)
        H = H.view(-1, 50 * 53 * 53)
        if debug:
            print("view shape:", H.shape)
        H = self.feature_extractor_part2(H)  # NxL
        H = torch.cat([H, c.repeat(H.size(0), 1)], dim=1)
        if debug:
            print("feature_extractor_part2 shape:", H.shape)

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, C, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X, C)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, C, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X, C)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        #print("Y_prob is：", Y_prob)
        return neg_log_likelihood, Y_prob, A

    def calculate_weights(self, X):
        Y_prob, Y_hat, weights = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        return Y_prob, Y_hat, weights