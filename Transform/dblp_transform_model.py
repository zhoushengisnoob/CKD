import numpy as np
from collections import defaultdict
import networkx as nx
data_folder, model_folder = '../Data', '../Model'
node_file, link_file, label_file = 'node.dat', 'link.dat', 'label.dat'
info_file, meta_file = 'info.dat', 'meta.dat'


def ckd_link_convert(dataset, attributed,version,use_target_node):
   
    def add_node(neigh_map,start_node,end_node,type_map,all_node_types):
        
        if start_node not in neigh_map:
            neigh_map[start_node]={}
            for node_type in all_node_types:
                neigh_map[start_node][node_type]=set()
        
        if end_node not in neigh_map:
            neigh_map[end_node]={}
            for node_type in all_node_types:
                neigh_map[end_node][node_type]=set()
        
        neigh_map[start_node][type_map[end_node]].add(end_node)
        neigh_map[end_node][type_map[start_node]].add(start_node)

    
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/CKD/data/{dataset}/{version}'
    node_type_map={}#node->node_type
    node_neigh_type_map={}
    node_types=set()
    target_node_set=set()
    node2id = {}
    useful_types=[]
    
    print(f'CKD: writing {dataset}\'s config file!')
    target_node, target_edge, ltypes = 0, 0, []

    with open(f'{ori_data_folder}/{info_file}','r') as original_info_file:
        for line in original_info_file:
            if line.startswith('Targeting: Link Type'): target_edge = int(line[:-2].split(',')[-1])

            if line.startswith('Targeting: Label Type'): target_node = int(line.split(' ')[-1])
    
    with open(f'{ori_data_folder}/{info_file}','r') as original_info_file:
        lstart = False
        
        for line in original_info_file:
            if line.startswith('LINK'):
                lstart=True
                continue
            if lstart and line[0]=='\n': break
            if lstart:
                line = line[:-1].split('\t')
                ltype, snode, enode, _ = list(filter(lambda x: len(x)!=0, line))
                ltypes.append((snode, enode, ltype))
    
    config_file = open(f'{model_data_folder}/config.dat','w')
    config_file.write(f'{target_node}\n')#1
    config_file.write(f'{target_edge}\n')#2
    config_file.write('{}\n'.format('\t'.join(list(map(lambda x: ','.join(x), ltypes)))))
    config_file.close()

    print('CKD Link: converting {}\'s node file for {} training!'.format(dataset,
                                                                    'attributed' if attributed == 'True' else 'unattributed'))
    
    new_node_file = open(f'{model_data_folder}/{node_file}', 'w')
    with open(f'{ori_data_folder}/{node_file}', 'r') as original_node_file:
        #cnt = 0
        for line in original_node_file:
            line = line[:-1].split()
            if attributed == 'True':
                new_node_file.write(f'{line[0]}\t{line[2]}\t{line[3]}\n')
            elif attributed == 'False':
                new_node_file.write(f'{line[0]}\t{line[2]}\n')
            
            node_type_map[int(line[0])]=int(line[2])
            node_types.add(int(line[2]))

            if int(line[2])==target_node:
                
                node2id[int(line[0])]=len(node2id)
                target_node_set.add(int(line[0]))
    new_node_file.close()

    print(f'CKD Link: converting {dataset}\'s label file')
    new_label_file = open(f'{model_data_folder}/{label_file}', 'w')
    
    with open(f'{ori_data_folder}/{label_file}', 'r') as original_label_file:
        for line in original_label_file:
            
            line = line[:-1].split('\t')
            new_label_file.write(f'{line[0]}\t{line[3]}\n')
    
    new_label_file.close()
    type_corners = {0: defaultdict(set)}
    print(f'CKD: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}', 'w')

    with open(f'{ori_data_folder}/{link_file}', 'r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left}\t{right}\t{ltype}\n')
            #add_node(node_neigh_type_map,int(left),int(right),node_type_map,node_types)
            # origin_graph.add_edge(int(left), int(right), weight=int(weight), ltype=int(ltype),
            #                direction=1 if left <= right else -1)
            start, end, ltype = int(left), int(right), int(ltype)
            if ltype == 0:
                type_corners[0][end].add(node2id[start])

    apa_lst = []
    for _, neighbors in type_corners[0].items():
        for snode in neighbors:
            for enode in neighbors:
                if snode!=enode:
                    #two_hops[snode].add(enode)
                    apa_lst.append([snode,enode])
    
    type_corners = {i: defaultdict(set) for i in [0,2,3,5]}


    #apvpa
    with open(f'{ori_data_folder}/{link_file}', 'r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left}\t{right}\t{ltype}\n')
            start, end, ltype = int(left), int(right), int(ltype)

            if ltype == 0:
                type_corners[0][end].add(node2id[start])

            if ltype == 2:
                type_corners[2][start].add(end)

            if ltype == 3:
                type_corners[3][start].add(node2id[end])

            if ltype == 5:
                type_corners[5][end].add(start)
    
    apv_lst = []
    for key, neighbors in type_corners[0].items():
        neighbors2 = type_corners[2][key]
        for snode in neighbors:
            for enode in neighbors2:
                apv_lst.append([snode, enode])
    
    dct = defaultdict(set)
    for tp in apv_lst:
        dct[tp[1]].add(tp[0])
    
    apvpa_lst = []
    for _, neighbors in dct.items():
        for snode in neighbors:
            for enode in neighbors:
                if snode!=enode:
                    apvpa_lst.append([snode,enode])

    type_corners = {i: defaultdict(set) for i in [0,1]}


    #aptpa
    with open(f'{ori_data_folder}/{link_file}', 'r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left}\t{right}\t{ltype}\n')
            start, end, ltype = int(left), int(right), int(ltype)

            if ltype == 0:
                type_corners[0][end].add(node2id[start])

            if ltype == 1:
                type_corners[1][start].add(end)

    apt_lst = []
    for key, neighbors in type_corners[0].items():
        neighbors2 = type_corners[1][key]
        for snode in neighbors:
            for enode in neighbors2:
                apt_lst.append([snode, enode])
   
    dct = defaultdict(set)
    for tp in apt_lst:
        dct[tp[1]].add(tp[0])
    
    aptpa_lst = []
    for _, neighbors in dct.items():
        for snode in neighbors:
            for enode in neighbors:
                if snode!=enode:
                    #two_hops[snode].add(enode)
                    aptpa_lst.append([snode,enode])


    #apa
    graph=nx.Graph(node_type=int)
    for tp in apa_lst:
        graph.add_edge(tp[0],tp[1])
    
    for node in node2id.values():
        if node not in graph:
            graph.add_edge(node,node)
    
    print(f'write graph apa,node:{len(graph.nodes)},edge:{len(graph.edges)}')
    nx.write_edgelist(graph,path=f"{model_data_folder}/sub_graph_0.edgelist",delimiter='\t',data=False)
    #nx.write_edgelist(graph,path=f"{model_data_folder}/sub_graph_apa.edgelist",delimiter='\t',data=False)
    
    #aptpa
    graph=nx.Graph(node_type=int)
    for tp in aptpa_lst:
        graph.add_edge(tp[0],tp[1])
    
    for node in node2id.values():
        if node not in graph:
            graph.add_edge(node,node)
    
    print(f'write graph aptpa,node:{len(graph.nodes)},edge:{len(graph.edges)}')
    nx.write_edgelist(graph,path=f"{model_data_folder}/sub_graph_1.edgelist",delimiter='\t',data=False)
    #nx.write_edgelist(graph,path=f"{model_data_folder}/sub_graph_aptpa.edgelist",delimiter='\t',data=False)

    #apvpa
    graph=nx.Graph(node_type=int)
    for tp in apvpa_lst:
        graph.add_edge(tp[0],tp[1])
    
    for node in node2id.values():
        if node not in graph:
            graph.add_edge(node,node)
    
    print(f'write graph apvpa,node:{len(graph.nodes)},edge:{len(graph.edges)}')
    nx.write_edgelist(graph,path=f"{model_data_folder}/sub_graph_2.edgelist",delimiter='\t',data=False)
    #nx.write_edgelist(graph,path=f"{model_data_folder}/sub_graph_apvpa.edgelist",delimiter='\t',data=False)

    with open(f"{model_data_folder}/node2id.txt",'w') as f:
        for node,id in node2id.items():
            f.write('\t'.join([str(node),str(id)])+'\n')





