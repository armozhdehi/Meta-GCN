# !pip install higher tensorboard scipy networkx scikit-learn
import torch
from tqdm.notebook import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets,transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import time
import torch.nn.functional as F
import copy
from torch import autograd
import higher
import itertools
import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import pickle
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import argparse
import scipy.sparse as sp
import numpy as np
import torch
import ipdb
from scipy.io import loadmat
import networkx as nx
import multiprocessing as mp
import torch.nn.functional as F
from functools import partial
import random
from sklearn.metrics import roc_auc_score, f1_score
from copy import deepcopy
from scipy.spatial.distance import pdist,squareform
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import ipdb
import copy
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import init
import argparse
import scipy.sparse as sp
import numpy as np
import torch
import ipdb
from scipy.io import loadmat
from collections import defaultdict
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import pickle
import torch.nn.functional as F
import itertools as it
import matplotlib.pyplot as plt
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import ipdb
import copy
import argparse
import scipy.sparse as sp
import numpy as np
import torch
import ipdb
from scipy.io import loadmat
import networkx as nx
import multiprocessing as mp
import torch.nn.functional as F
from functools import partial
import random
from sklearn.metrics import roc_auc_score, f1_score
from copy import deepcopy
from scipy.spatial.distance import pdist,squareform
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import init
import ipdb
import argparse
import scipy.sparse as sp
import numpy as np
import torch
import ipdb
from scipy.io import loadmat
from collections import defaultdict
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True


# class GraphConvolution(Module):
#     def __init__(self, in_features, out_features, order, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.order = order
#         self.weight = torch.nn.ParameterList([])
#         for i in range(self.order):
#             self.weight.append(Parameter(torch.FloatTensor(in_features, out_features)))
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         for i in range(self.order):
#             stdv = 1. / math.sqrt(self.weight[i].size(1))
#             self.weight[i].data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)

#     def forward(self, input, adj_mtx):
#         output = []
#         if self.order == 1 and type(adj_mtx) != list:
#             adj = [adj_mtx]
#         for i in range(self.order):
#             support = torch.mm(input, self.weight[i])
#             # output.append(support)
#             output.append(torch.mm(adj_mtx[i], support))
#         output = sum(output)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'


# GCN layer based on : https://arxiv.org/abs/1609.02907
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias = True, init = None):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.DoubleTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.DoubleTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(init)

    def reset_parameters(self, init):
        stdv = 1. / math.sqrt(self.weight.size(1))
        if init == "Xavier Uniform":
            nn.init.xavier_uniform_(self.weight.data)
        elif init == "Xavier Normal":
            nn.init.xavier_normal_(self.weight.data)
        elif init == "Kaiming Uniform":
            nn.init.kaiming_uniform_(self.weight.data)
        elif init == "Kaiming Normal":
            nn.init.kaiming_normal_(self.weight.data)
        else:
            self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        #for 3_D batch, need a loop!!!


        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN_Encoder_s(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout, init = None):
        super(GCN_Encoder_s, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, init = init)
        self.dropout = dropout

    def forward(self, x, adj, funct):
        if funct == "ReLU":
            func = F.relu
        elif funct == "LeakyReLU":
            func = F.leaky_relu
        elif funct == "Sigmoid":
            func = sigmoid
        elif funct == "Sigmoid":
            func = nn.Sigmoid()
        elif funct == "PReLU":
            func = nn.PReLU()
        x = func(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x

class GCN_Classifier_s(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout, device, init = None):
        super(GCN_Classifier_s, self).__init__()

        self.gc1 = GraphConvolution(nembed, nhid, init = init)
        self.mlp = nn.Linear(nhid, nclass, device = device).double()
        self.dropout = dropout

        self.reset_parameters(init)

    def reset_parameters(self, init):
        if init == "Xavier Uniform":
            nn.init.xavier_uniform_(self.mlp.weight.data)
            # nn.init.xavier_uniform_(self.bias.data)
        elif init == "Xavier Normal":
            nn.init.xavier_normal_(self.mlp.weight.data)
        elif init == "Kaiming Uniform":
            nn.init.kaiming_uniform_(self.mlp.weight.data)
        elif init == "Kaiming Normal":
            nn.init.kaiming_normal_(self.mlp.weight.data)
        else:
            nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x, adj, funct):
        if funct == "ReLU":
            func = F.relu
        elif funct == "LeakyReLU":
            func = F.leaky_relu
        elif funct == "Sigmoid":
            func = sigmoid
        elif funct == "Sigmoid":
            func = nn.Sigmoid()
        elif funct == "PReLU":
            func = nn.PReLU()
        x = func(self.gc1(x, adj)).double()
        x = F.dropout(x, self.dropout, training=self.training).double()
        x = self.mlp(x).double()

        return x

class GCN_Encoder_w(nn.Module):
    def __init__(self, nfeat, nclass, nhid, nembed, dropout, device, func, init = None):
        super(GCN_Encoder_w, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, init = init)
        self.dropout = dropout
        self.gc2 = GraphConvolution(nembed, nhid, init = init)
        self.mlp = nn.Linear(nhid, nclass, device = device).double()
    def forward(self, x, adj, funct):
        if funct == "ReLU":
            func = F.relu
        elif funct == "LeakyReLU":
            func = F.leaky_relu
        elif funct == "Sigmoid":
            func = sigmoid
        elif funct == "Sigmoid":
            func = nn.Sigmoid()
        elif funct == "PReLU":
            func = nn.PReLU()
        x = func(self.gc1(x, adj)).double()
        x = F.dropout(x, self.dropout, training=self.training).double()
        x = func(self.gc2(x, adj)).double()
        x = F.dropout(x, self.dropout, training=self.training).double()
        x = self.mlp(x).double()
        return x

class Decoder_s(Module):
    """
    Simple Graphsage layer
    """

    def __init__(self, nembed, dropout=0.1, init = None):
        super(Decoder_s, self).__init__()
        self.dropout = dropout

        self.de_weight = Parameter(torch.FloatTensor(nembed, nembed))
        # if init == "Xavier":
        #     nn.init.kaiming_uniform_(self.weight.data)
        # else:
        
        self.reset_parameters(init)


    def reset_parameters(self, init):
        if init == "Xavier Uniform":
            nn.init.xavier_uniform_(self.de_weight.data)
            # nn.init.xavier_uniform_(self.bias.data)
        elif init == "Xavier Normal":
            nn.init.xavier_normal_(self.de_weight.data)
        elif init == "Kaiming Uniform":
            nn.init.kaiming_uniform_(self.de_weight.data)
        elif init == "Kaiming Normal":
            nn.init.kaiming_normal_(self.de_weight.data)
        else:
            stdv = 1. / math.sqrt(self.de_weight.size(1))
            self.de_weight.data.uniform_(-stdv, stdv)


    def forward(self, node_embed):
        
        combine = F.linear(node_embed, self.de_weight)
        adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1,-2)))

        return adj_out


class GCN_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout, init = None):
        super(GCN_Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, init = init)
        self.dropout = dropout

    def forward(self, x, adj, funct):
        if funct == "ReLU":
            func = F.relu
        elif funct == "LeakyReLU":
            func = F.leaky_relu
        elif funct == "Sigmoid":
            func = sigmoid
        elif funct == "Sigmoid":
            func = nn.Sigmoid().double()
        elif funct == "PReLU":
            func = nn.PReLU().double()
        x = func(self.gc1(x, adj)).double()
        x = F.dropout(x, self.dropout, training=self.training).double()

        return x

class GCN_Encoder2(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout, init = None):
        super(GCN_Encoder2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, init = init)
        self.gc2 = GraphConvolution(nhid, nembed, init = init)
        self.dropout = dropout

    def forward(self, x, adj, funct):
        if funct == "ReLU":
            func = F.relu
        elif funct == "LeakyReLU":
            func = F.leaky_relu
        elif funct == "Sigmoid":
            func = sigmoid
        elif funct == "Sigmoid":
            func = nn.Sigmoid().double()
        elif funct == "PReLU":
            func = nn.PReLU().double()
        x = func(self.gc1(x, adj)).double()
        x = F.dropout(x, self.dropout, training=self.training).double()
        x = func(self.gc2(x, adj)).double()
        x = F.dropout(x, self.dropout, training=self.training).double()
        return x

class GCN_Encoder3(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout, nclass, order, init = None):
        super(GCN_Encoder3, self).__init__()

        layers = []
        if len(nhid) == 0:
            # layers.append(GraphConvolution(nfeat, nclass, order=order))
            layers.append(GraphConvolution(nfeat, nclass, init = init))
        else:
            # layers.append(GraphConvolution(nfeat, nhid[0], order=order))
            layers.append(GraphConvolution(nfeat, nhid[0], init = init))
            for i in range(len(nhid) - 1):
                # layers.append(GraphConvolution(nhid[i], nhid[i + 1], order=order))
                layers.append(GraphConvolution(nhid[i], nhid[i + 1], init = init))
        if nclass > 1:
            layers.append(GraphConvolution(nhid[-1], nembed, init = init))
            # layers.append(GraphConvolution(nhid[-1], nclass, order=order))
        self.gc = nn.ModuleList(layers)

        self.dropout = dropout
        self.nclass = nclass

    def forward(self, x, adj, funct):
        if funct == "ReLU":
            func = F.relu
        elif funct == "LeakyReLU":
            func = F.leaky_relu
        elif funct == "Sigmoid":
            func = sigmoid
        elif funct == "Sigmoid":
            func = nn.Sigmoid().double()
        elif funct == "PReLU":
            func = nn.PReLU().double()
        end_layer = len(self.gc) - 1 if self.nclass > 1 else len(self.gc)
        for i in range(end_layer):
            x = F.dropout(x, self.dropout, training=self.training).double()
            x = self.gc[i](x, adj).double()
            x = func(x).double()
        return x

class GCN_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout, device, init = None):
        super(GCN_Classifier, self).__init__()
        self.device = device
        self.gc1 = GraphConvolution(nembed, nhid, init = init)
        self.mlp = nn.Linear(nhid, nclass, device=device).double()
        # self.mlp = nn.Linear(nhid, nclass, device=device)
        self.dropout = dropout

        self.reset_parameters(init)

    def reset_parameters(self, init):
        if init == "Xavier Uniform":
            nn.init.xavier_uniform_(self.mlp.weight.data)
            # nn.init.xavier_uniform_(self.bias.data)
        elif init == "Xavier Normal":
            nn.init.xavier_normal_(self.mlp.weight.data)
        elif init == "Kaiming Uniform":
            nn.init.kaiming_uniform_(self.mlp.weight.data)
        elif init == "Kaiming Normal":
            nn.init.kaiming_normal_(self.mlp.weight.data)
        else:
            nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x, adj, funct):
        if funct == "ReLU":
            func = F.relu
        elif funct == "LeakyReLU":
            func = F.leaky_relu
        elif funct == "Sigmoid":
            func = sigmoid
        elif funct == "Sigmoid":
            func = nn.Sigmoid().double()
        elif funct == "PReLU":
            func = nn.PReLU().double()
        x = func(self.gc1(x, adj)).double()
        x = F.dropout(x, self.dropout, training=self.training).double()
        x = self.mlp(x).double()

        return x

# Sigmoid function
def sigmoid(mx):
    mx = torch.sigmoid(mx).double()
    return F.normalize(mx, p=1).double()