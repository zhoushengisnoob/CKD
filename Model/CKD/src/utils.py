import numpy as np
import pandas as pd
import networkx as nx
import os
import random
import torch
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
import math
import heapq
from datetime import datetime


def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)



def PPR(node2neigh_list,alpha=0.15):
    sim_matrix_list=[]
    adjs=[]
    for node2neigh in node2neigh_list:
        adj=[]
        for node in range(len(node2neigh)):
            neighs=np.zeros(shape=len(node2neigh),dtype=np.int32)
            neighs[node2neigh[node]]=1
            neighs[node]=1
            adj.append(neighs.reshape(1,-1))

        #adjs.append()
        adj=np.concatenate(adj,axis=0).astype(np.float32)
        I=np.diag(np.ones(shape=len(node2neigh)))
        adjs.append(adj)
        sim_matrix_list.append(alpha*np.linalg.inv((I-(1-alpha)*(adj/np.sum(adj,axis=1).reshape(1,-1)))).astype(np.float32))
    return adjs,sim_matrix_list



def find_topk(pairs,max_k):
    result=[]
    for pair in pairs:
        if len(result)<max_k:
            heapq.heappush(result,pair)
            continue
        if result[0][0]<pair[0]:
            heapq.heappop(result)
            heapq.heappush(result,pair)
    result.sort(reverse=True)
    return [pair[1] for pair in result]


def get_topk_neigh(adjs,node,topk,node2neigh_list):
    """
    Get the node TOPK neighbors and rebuild the subgraph
    """
    result=[]
    for idx,adj in enumerate(adjs):
        node2neigh = node2neigh_list[idx]
        pairs=[(adj[i],i) for i in range(len(adj))]
        pairs[node]=(-100,node)
        if len(node2neigh[node])<=1:
            topk_nodes = [node] * (topk+1)
        else:
            topk_nodes = [node] + find_topk(pairs, topk)

        result.append(topk_nodes)
    return result

def get_topk_neigh_single(data):
    adjs, node, topk, node2neigh_list=data
    return get_topk_neigh(adjs,node,topk,node2neigh_list)

def get_topk_adj_transfer(new_adjs):
    result=[]
    for channel_adj in new_adjs:
        D=1/np.sqrt(np.sum(channel_adj,axis=1))
        channel_adj=channel_adj*D.reshape(1,-1)*D.reshape(-1,1)
        channel_adj=channel_adj.reshape(1,channel_adj.shape[0],channel_adj.shape[1]).astype(np.float32)
        result.append(channel_adj)
    return result

def get_topk_neigh_multi(target_nodes,node2neigh_list,topk,adjs,Ss):

    data=[]
    for node in range(len(target_nodes)):
        data.append([[adj[node] for adj in Ss],node,topk,node2neigh_list])
    pool=Pool(10)
    topk_result=pool.map(get_topk_neigh_single,data)
    pool.close()
    pool.join()

    print('sample topk neigh finish:',datetime.now())
    new_adjs=[]
    for topk_list in (topk_result):
        temp=[]
        for idx,topk_neigh in enumerate(topk_list):
            new_adj=adjs[idx][topk_neigh][:,topk_neigh]
            temp.append(new_adj)
        new_adjs.append(temp)

    adj_result=[]
    for new_adj in new_adjs:
        adj_result.append(get_topk_adj_transfer(new_adj))

    final_result=[]
    for idx in range(len(topk_result)):
        temp=[]
        status=[]
        for idy in range(len(topk_result[idx])):
            node2neigh=node2neigh_list[idy]
            status.append(0 if len(node2neigh[idx])<=1 else 1)
            temp.append([np.array(topk_result[idx][idy]),adj_result[idx][idy]])
        final_result.append([idx,status,temp])
    return final_result



def load_node2new_id(path):
    """
    Read the mapping from the original ID to the new ID of the target node. The range of the new ID is (0, len (nodes) - 1)
    """
    name2id={}
    with open(path) as f:
        for line in f:
            if not line:
                continue
            data=[int(i) for i in line.strip().split('\t')]
            name2id[data[0]]=data[1]
    return name2id

def load_sub_graph(path,name2id):
    """
    Read different neighbors to build subgraphs,The node ID in the subgraph has been changed to the new ID form in advance
    """
    node2neigh={}
    with open(path) as f:
        for line in f:
            snode,enode=[int(i) for i in line.strip().split('\t')]
            if snode not in node2neigh:
                node2neigh[snode]=set()
            if enode not in node2neigh:
                node2neigh[enode]=set()
            node2neigh[snode].add(enode)
            node2neigh[enode].add(snode)
    return node2neigh

def load_node(path):
    """
    Load the old ID of the node and the characteristics of the node. If the node has no characteristics, 
    it will be initialized directly at random
    """
    node2attr={}
    with open(path) as f:
        for line in f:
            data=line.strip().split('\t')
            node2attr[int(data[0])]=np.array([float(i) for i in data[2].split(',')],dtype=np.float32)
    return node2attr

def load_data(ltypes,base_path,use_features):

    name2id=load_node2new_id(os.path.join(base_path,'node2id.txt'))
    id2name={v:k for k,v in name2id.items()}
    node2neigh_list=[]
    pos_edges=set()
    print(ltypes)
    for ltype in ltypes:
        graph=load_sub_graph(os.path.join(base_path,f'sub_graph_{ltype}.edgelist'),name2id)

        node2neigh_list.append(graph)

    features = None
    if use_features:
        features=[]
        name2attr=load_node(path=os.path.join(base_path,'node.dat'))
        for i in range(len(name2id)):
            origin_id=id2name[i]
            feature=name2attr[origin_id]
            features.append(feature)
        features=np.array(features,dtype=np.float32)
    node2neigh_pos={}
    for node2neigh in node2neigh_list:
        for k,v in node2neigh.items():
            if k not in node2neigh_pos:
                node2neigh_pos[k]=[]
            node2neigh[k]=list(v)
            for neigh_node in v:
                node2neigh_pos[k].append(neigh_node)
                if k!=neigh_node:
                    pos_edges.add((k,neigh_node))
    return name2id,id2name,features,node2neigh_list,list(pos_edges)



def load_link_test_data(dataset,emb_list,name2id,need_handle=True):
    path=f'../../../Data/{dataset}/link.dat.test'
    test_graph=nx.read_edgelist(path,delimiter='\t',nodetype=int,data=(('weight',int),))
    print('in load data')
    if need_handle:
        for idx,emb in enumerate(emb_list):
            if type(emb) != np.ndarray:
                emb_list[idx]=emb.cpu().detach().numpy()
    sum_emb=sum(emb_list)
    concat_emb=np.concatenate(emb_list,axis=1)
    emb_list+=[sum_emb,concat_emb]
    sum_auc=0
    for idx,emb in enumerate(emb_list[-2:]):
        y_true = []
        y_pred = []
        for edge, weight in test_graph.edges().items():
            y_true.append(int(weight['weight']))
            pred_prob = np.dot(emb[name2id[edge[0]]], emb[name2id[edge[1]]])
            y_pred.append(pred_prob)

        auc=roc_auc_score(y_true, y_pred)
        print(f'link prediction auc:{auc}')
        if idx==0:
            sum_auc=auc
    return sum_auc


def cls_test(dataset,emb_list,name2id):
    base_path = f'../../../Data/{dataset}/'
    label_file_path=f'{base_path}label.dat'
    label_test_path=f'{base_path}label.dat.test'
    seed = 1
    max_iter = 3000
    print('in load data')
    for emb in emb_list:
        labels, embeddings = [], []
        for file_path in [label_file_path, label_test_path]:
            with open(file_path, 'r') as label_file:
                for line in label_file:
                    index, _, _, label = line[:-1].split('\t')
                    labels.append(label)
                    embeddings.append(emb[name2id[int(index)]])
        labels, embeddings = np.array(labels).astype(int), np.array(embeddings)

        macro, micro = [], []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(embeddings, labels):
            clf = LinearSVC(random_state=seed, max_iter=max_iter)
            clf.fit(embeddings[train_idx], labels[train_idx])
            preds = clf.predict(embeddings[test_idx])

            macro.append(f1_score(labels[test_idx], preds, average='macro'))
            micro.append(f1_score(labels[test_idx], preds, average='micro'))

        print('cls:', np.mean(macro), np.mean(micro))

def output(args, embeddings, id2name,need_handle=True):
    print('output data')
    if need_handle:
        for idx, emb in enumerate(embeddings):
            if type(emb)!=np.ndarray:
                embeddings[idx] = emb.cpu().detach().numpy()
    embeddings = sum(embeddings)
    output_path=f'../data/{args.dataset}/{args.output}'
    with open(output_path, 'w') as file:
        file.write(
            f'size={args.size}, dropout={args.dropout}, ,topk:{args.topk}, lr={args.lr}, batch-size={args.batch_size}, epochs={args.epochs}, attributed={args.attributed}, supervised={args.supervised}\n')
        for nid, name in id2name.items():
            
            file.write('{}\t{}\n'.format(name, ' '.join([str(i) for i in embeddings[nid]])))