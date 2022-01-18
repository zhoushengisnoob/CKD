import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanAggregator(nn.Module):
    def __init__(self, dim,activation=None):
        super(MeanAggregator,self).__init__()
        self.dim=dim
        self.self_W= nn.Parameter(torch.zeros(size=(dim, dim//2)))
        nn.init.xavier_uniform_(self.self_W.data)
        self.neigh_W = nn.Parameter(torch.zeros(size=(dim, dim // 2)))
        nn.init.xavier_uniform_(self.neigh_W.data)
        self.activate=activation

    def forward(self,self_emb,neigh_emb):
        agg_emb=torch.mean(neigh_emb,dim=1)
        from_self=torch.matmul(self_emb,self.self_W)
        from_neigh = torch.matmul(agg_emb,self.neigh_W)
        if self.activate:
            from_self = self.activate(from_self)
            from_neigh=self.activate(from_neigh)

        return torch.cat([from_self,from_neigh],dim=1)


class SageEncoder(nn.Module):
    def __init__(self,nlayer,feature_dim,alpha,dim,fanouts):
        super(SageEncoder,self).__init__()
        self.nlayer=nlayer
        self.aggregator=[]
        for layer in range(self.nlayer):
            activation=nn.ReLU() if layer<self.nlayer-1 else None
            mean_aggregator=MeanAggregator(dim,activation=activation).cuda()
            self.aggregator.append(mean_aggregator)
            self.add_module(f'mean_aggregator_{layer}',mean_aggregator)
        self.dims=[feature_dim]+[dim]*self.nlayer
        self.fanouts=fanouts

    def sample(self,features,sample_nodes):

        feature_list=[]
        for sample_node_list in sample_nodes:
            feature_list.append(features[sample_node_list,:])

        return feature_list

    def forward(self,features,sample_nodes):
        hidden=self.sample(features,sample_nodes)
        for layer in range(self.nlayer):
            aggregator=self.aggregator[layer]
            next_hidden=[]
            for hop in range(self.nlayer-layer):
                neigh_shape=[-1,self.fanouts[hop],self.dims[layer]]
                h=aggregator(hidden[hop],torch.reshape(hidden[hop+1],neigh_shape))
                next_hidden.append(h)
            hidden=next_hidden

        return hidden[0]


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act='prelu', bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act


        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        if self.act is not None:
            out=self.act(out)
            #out = torch.squeeze(out, 0)
        return out


class CKD(nn.Module):
    def __init__(self, in_ft, out_ft, layers,act='prelu', bias=True,idx=0):
        super(CKD, self).__init__()
        self.layers=layers
        self.in_ft=in_ft
        self.out_ft=out_ft
        self.gcn_list=[]
        self.dim=[in_ft]+[out_ft]*layers

        self.node_trans = torch.nn.Parameter(torch.zeros(size=(self.out_ft, self.out_ft)))
        nn.init.xavier_uniform_(self.node_trans.data)
        self.graph_trans = torch.nn.Parameter(torch.zeros(size=(self.out_ft, self.out_ft)))
        nn.init.xavier_uniform_(self.graph_trans.data)

        for layer in range(self.layers):
            gcn=GCN(self.dim[layer],self.dim[layer+1],act='prelu' if layer!=self.layers-1 else None)
            self.gcn_list.append(gcn)
            self.add_module(f'gcn_{idx}_{layer}',gcn)

    def readout(self, node_emb):
        return torch.sigmoid(torch.mean(node_emb, dim=1, keepdim=True).squeeze(dim=1))

    def forward(self,seq,adj):
        out=seq
        for layer in range(self.layers):
            gcn=self.gcn_list[layer]
            out=gcn(out,adj)
        graph_emb=self.readout(out)
        return out,graph_emb
