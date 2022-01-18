import numpy as np
from collections import defaultdict
import networkx as nx
data_folder, model_folder = '../Data', '../Model'
node_file, link_file, label_file = 'node.dat', 'link.dat', 'label.dat'
info_file, meta_file = 'info.dat', 'meta.dat'


def metapath2vec_esim_convert(dataset):
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/metapath2vec-ESim/data/{dataset}'

    print(f'metapath2vec-ESim: converting {dataset}\'s node file!')
    new_node_file = open(f'{model_data_folder}/{node_file}', 'w')
    with open(f'{ori_data_folder}/{node_file}', 'r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
            new_node_file.write(f'{line[0]} {line[2]}\n')
    new_node_file.close()

    print(f'metapath2vec-ESim: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}', 'w')
    with open(f'{ori_data_folder}/{link_file}', 'r') as original_link_file:
        for line in original_link_file:
            left, right, _, _ = line[:-1].split('\t')
            new_link_file.write(f'{left} {right}\n')
    new_link_file.close()

    print(f'metapath2vec-ESim: writing {dataset}\'s path file!')
    next_node = defaultdict(list)
    with open(f'{ori_data_folder}/{info_file}', 'r') as original_info_file:
        start = False
        for line in original_info_file:
            if line[:4] == 'LINK':
                start = True
                continue
            if start and line[0] == '\n':
                break
            if start:
                line = line[:-1].split('\t')
                ltype, snode, enode, _ = list(filter(lambda x: len(x) != 0, line))
                next_node[snode].append(enode)
    with open(f'{model_data_folder}/path.dat', 'w') as new_path_file:
        for start, ends in next_node.items():
            for end in ends:
                new_path_file.write(f'{start}{end} 1.0\n')
                if end in next_node:
                    for twohop in next_node[end]:
                        new_path_file.write(f'{start}{end}{twohop} 0.5\n')

    return


def pte_convert(dataset):
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/PTE/data/{dataset}'

    print(f'PTE: converting {dataset}\'s node file!')
    new_node_file = open(f'{model_data_folder}/{node_file}', 'w')
    with open(f'{ori_data_folder}/{node_file}', 'r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
            new_node_file.write(f'{line[0]}\n')
    new_node_file.close()

    print(f'PTE: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}', 'w')
    with open(f'{ori_data_folder}/{link_file}', 'r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left} {right} {ltype} {weight}\n')
    new_link_file.close()

    print(f'PTE: writing {dataset}\'s type file!')
    type_count = 0
    with open(f'{ori_data_folder}/{meta_file}', 'r') as original_meta_file:
        for line in original_meta_file:
            entity, info, _, _ = line[:-1].split(' ')
            info = info[:-1].split('_')
            if entity == 'Edge' and info[0] == 'Type':
                type_count += 1
    new_type_file = open(f'{model_data_folder}/type.dat', 'w')
    new_type_file.write(f'{type_count}\n')
    new_type_file.close()

    return


def hin2vec_convert(dataset):
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/HIN2Vec/data/{dataset}'

    print(f'HIN2Vec: reading {dataset}\'s node file!')
    type_dict = {}
    with open(f'{ori_data_folder}/{node_file}', 'r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
            type_dict[line[0]] = line[2]

    print(f'HIN2Vec: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}', 'w')
    with open(f'{ori_data_folder}/{link_file}', 'r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left}\t{type_dict[left]}\t{right}\t{type_dict[right]}\t{ltype}\n')
    new_link_file.close()

    return


def aspem_convert(dataset):
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/AspEm/data/{dataset}'

    print(f'AspEm: converting {dataset}\'s node file!')
    type_dict = {}
    new_node_file = open(f'{model_data_folder}/{node_file}', 'w')
    with open(f'{ori_data_folder}/{node_file}', 'r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
            type_dict[line[0]] = line[2]
            new_node_file.write(f'{line[2]}:{line[0]} {line[2]}\n')
    new_node_file.close()

    print(f'AspEm: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}', 'w')
    with open(f'{ori_data_folder}/{link_file}', 'r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(
                f'{type_dict[left]}:{left} {type_dict[left]} {type_dict[right]}:{right} {type_dict[right]} {weight} {ltype}\n')
    new_link_file.close()

    print(f'AspEm: writing {dataset}\'s type file!')
    type_count, target_type = 0, -1
    with open(f'{ori_data_folder}/{meta_file}', 'r') as original_meta_file:
        for line in original_meta_file:
            entity, info, _, _ = line[:-1].split(' ')
            info = info[:-1].split('_')
            if entity == 'Node' and info[0] == 'Type':
                type_count += 1
            if entity == 'Label' and info[0] == 'Class':
                target_type = info[1]
                break
    new_type_file = open(f'{model_data_folder}/type.dat', 'w')
    new_type_file.write(f'{target_type}\n')
    new_type_file.write(f'{type_count}\n')
    new_type_file.close()

    return


def heer_convert(dataset):
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/HEER/data/{dataset}'

    print(f'HEER: reading {dataset}\'s node file!')
    type_dict, types = {}, set()
    with open(f'{ori_data_folder}/{node_file}', 'r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
            type_dict[line[0]] = line[2]
            types.add(line[2])

    print(f'HEER: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}', 'w')
    with open(f'{ori_data_folder}/{link_file}', 'r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{type_dict[left]}:{left} {type_dict[right]}:{right} {weight} {ltype}:d\n')
    new_link_file.close()

    print(f'HEER: writing {dataset}\'s config file!')
    edge_info = []
    with open(f'{ori_data_folder}/{info_file}', 'r') as original_info_file:
        start = False
        for line in original_info_file:
            if line[:4] == 'LINK':
                start = True
                continue
            if start and line[0] == '\n':
                break
            if start:
                line = line[:-1].split('\t')
                ltype, snode, enode, _ = list(filter(lambda x: len(x) != 0, line))
                edge_info.append([ltype, snode, enode])
    edge_info = np.array(edge_info).astype(int)
    with open(f'{model_data_folder}/config.dat', 'w') as new_config_file:
        new_config_file.write(f'{edge_info[:, 1:].tolist()}\n')
        new_config_file.write(f'{np.arange(len(types)).astype(str).tolist()}\n')
        temp = list(map(lambda x: f'{x}:d', edge_info[:, 0].tolist()))
        new_config_file.write(f'{temp}\n')
        new_config_file.write(f'{np.ones(len(edge_info)).astype(int).tolist()}\n')

    return


def rgcn_convert(dataset, attributed, supervised):
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/R-GCN/data/{dataset}'

    entity_count, relation_count = 0, 0
    with open(f'{ori_data_folder}/{meta_file}', 'r') as original_meta_file:
        for line in original_meta_file:
            entity, info, _, count = line[:-1].split(' ')
            info = info[:-1].split('_')
            if entity == 'Node' and info[0] == 'Total':
                entity_count = int(count)
            elif entity == 'Edge' and info[0] == 'Type':
                relation_count += 1

    print(f'R-GCN: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}', 'w')
    new_link_file.write(f'{entity_count} {relation_count}\n')
    with open(f'{ori_data_folder}/{link_file}', 'r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left} {ltype} {right}\n')
    new_link_file.close()

    if attributed == 'True':
        print(f'R-GCN: converting {dataset}\'s node file for attributed training!')
        new_node_file = open(f'{model_data_folder}/{node_file}', 'w')
        with open(f'{ori_data_folder}/{node_file}', 'r') as original_node_file:
            for line in original_node_file:
                line = line[:-1].split('\t')
                new_node_file.write(f'{line[0]}\t{line[3]}\n')
        new_node_file.close()

    if supervised == 'True':
        print(f'R-GCN: converting {dataset}\'s label file for semi-supervised training!')
        new_label_file = open(f'{model_data_folder}/{label_file}', 'w')
        with open(f'{ori_data_folder}/{label_file}', 'r') as original_label_file:
            for line in original_label_file:
                line = line[:-1].split('\t')
                new_label_file.write(f'{line[0]}\t{line[3]}\n')
        new_label_file.close()

    return


def han_convert(dataset, attributed, supervised):
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/HAN/data/{dataset}'

    print('HAN: converting {}\'s node file for {} training!'.format(dataset,
                                                                    'attributed' if attributed == 'True' else 'unattributed'))
    new_node_file = open(f'{model_data_folder}/{node_file}', 'w')
    with open(f'{ori_data_folder}/{node_file}', 'r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
            if attributed == 'True':
                new_node_file.write(f'{line[0]}\t{line[2]}\t{line[3]}\n')
            elif attributed == 'False':
                new_node_file.write(f'{line[0]}\t{line[2]}\n')
    new_node_file.close()

    print(f'HAN: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}', 'w')
    with open(f'{ori_data_folder}/{link_file}', 'r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left}\t{right}\t{ltype}\n')
    new_link_file.close()

    print(f'HAN: writing {dataset}\'s config file!')
    target_node, target_edge, ltypes = 0, 0, []
    with open(f'{ori_data_folder}/{info_file}', 'r') as original_info_file:
        for line in original_info_file:
            if line.startswith('Targeting: Link Type'): target_edge = int(line[:-2].split(',')[-1])
            if line.startswith('Targeting: Label Type'): target_node = int(line.split(' ')[-1])
    with open(f'{ori_data_folder}/{info_file}', 'r') as original_info_file:
        lstart = False
        for line in original_info_file:
            if line.startswith('LINK'):
                lstart = True
                continue
            if lstart and line[0] == '\n': break
            if lstart:
                line = line[:-1].split('\t')
                ltype, snode, enode, _ = list(filter(lambda x: len(x) != 0, line))
                ltypes.append((snode, enode, ltype))
    config_file = open(f'{model_data_folder}/config.dat', 'w')
    config_file.write(f'{target_node}\n')
    config_file.write(f'{target_edge}\n')
    config_file.write('{}\n'.format('\t'.join(list(map(lambda x: ','.join(x), ltypes)))))
    config_file.close()

    if supervised == 'True':
        print(f'HAN: converting {dataset}\'s label file for semi-supervised training!')
        new_label_file = open(f'{model_data_folder}/{label_file}', 'w')
        with open(f'{ori_data_folder}/{label_file}', 'r') as original_label_file:
            for line in original_label_file:
                line = line[:-1].split('\t')
                new_label_file.write(f'{line[0]}\t{line[3]}\n')
        new_label_file.close()

    return


def magnn_convert(dataset, attributed, supervised):
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/MAGNN/data/{dataset}'

    print('MAGNN: converting {}\'s node file for {} training!'.format(dataset,
                                                                      'attributed' if attributed == 'True' else 'unattributed'))
    new_node_file = open(f'{model_data_folder}/{node_file}', 'w')
    with open(f'{ori_data_folder}/{node_file}', 'r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
            if attributed == 'True':
                new_node_file.write(f'{line[0]}\t{line[2]}\t{line[3]}\n')
            elif attributed == 'False':
                new_node_file.write(f'{line[0]}\t{line[2]}\n')
    new_node_file.close()

    print(f'MAGNN: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}', 'w')
    with open(f'{ori_data_folder}/{link_file}', 'r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left}\t{right}\t{ltype}\n')
    new_link_file.close()

    print(f'MAGNN: writing {dataset}\'s path file!')
    next_node = defaultdict(list)
    with open(f'{ori_data_folder}/{info_file}', 'r') as original_info_file:
        start = False
        for line in original_info_file:
            if line[:4] == 'LINK':
                start = True
                continue
            if start and line[0] == '\n':
                break
            if start:
                line = line[:-1].split('\t')
                ltype, snode, enode, _ = list(filter(lambda x: len(x) != 0, line))
                next_node[snode].append(enode)
    with open(f'{model_data_folder}/path.dat', 'w') as new_path_file:
        for start, ends in next_node.items():
            for end in ends:
                new_path_file.write(f'{start}\t{end}\n')
                if end in next_node:
                    for twohop in next_node[end]:
                        new_path_file.write(f'{start}\t{end}\t{twohop}\n')

    if supervised == 'True':
        print(f'MAGNN: converting {dataset}\'s label file for semi-supervised training!')
        new_label_file = open(f'{model_data_folder}/{label_file}', 'w')
        with open(f'{ori_data_folder}/{label_file}', 'r') as original_label_file:
            for line in original_label_file:
                line = line[:-1].split('\t')
                new_label_file.write(f'{line[0]}\t{line[3]}\n')
        new_label_file.close()

    return


def hgt_convert(dataset, attributed, supervised):
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/HGT/data/{dataset}'

    print('HGT: converting {}\'s node file for {} training!'.format(dataset,
                                                                    'attributed' if attributed == 'True' else 'unattributed'))
    new_node_file = open(f'{model_data_folder}/{node_file}', 'w')
    with open(f'{ori_data_folder}/{node_file}', 'r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
            if attributed == 'True':
                new_node_file.write(f'{line[0]}\t{line[2]}\t{line[3]}\n')
            elif attributed == 'False':
                new_node_file.write(f'{line[0]}\t{line[2]}\n')
    new_node_file.close()

    print(f'HGT: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}', 'w')
    with open(f'{ori_data_folder}/{link_file}', 'r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left}\t{right}\t{ltype}\n')
    new_link_file.close()

    if supervised == 'True':
        print(f'HGT: converting {dataset}\'s label file for semi-supervised training!')
        labeled_type, nlabel, begin = None, -1, False
        with open(f'{ori_data_folder}/{info_file}', 'r') as file:
            for line in file:
                if line.startswith('Targeting: Label Type'):
                    labeled_type = int(line.split(' ')[-1])
                elif line == 'TYPE\tCLASS\tMEANING\n':
                    begin = True
                elif begin:
                    nlabel += 1
        new_label_file = open(f'{model_data_folder}/{label_file}', 'w')
        new_label_file.write(f'{labeled_type}\t{nlabel}\n')
        with open(f'{ori_data_folder}/{label_file}', 'r') as original_label_file:
            for line in original_label_file:
                line = line[:-1].split('\t')
                new_label_file.write(f'{line[0]}\t{line[3]}\n')
        new_label_file.close()

    return


def transe_convert(dataset):
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/TransE/data/{dataset}'

    entity_count, relation_count, triplet_count = 0, 0, 0
    with open(f'{ori_data_folder}/{meta_file}', 'r') as original_meta_file:
        for line in original_meta_file:
            entity, info, _, count = line[:-1].split(' ')
            info = info[:-1].split('_')
            if entity == 'Node' and info[0] == 'Total':
                entity_count = int(count)
            elif entity == 'Edge' and info[0] == 'Total':
                triplet_count = int(count)
            elif entity == 'Edge' and info[0] == 'Type':
                relation_count += 1

    print(f'TransE: converting {dataset}\'s node file!')
    new_node_file = open(f'{model_data_folder}/{node_file}', 'w')
    new_node_file.write(f'{entity_count}\n')
    with open(f'{ori_data_folder}/{node_file}', 'r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
            new_node_file.write(f'{line[0]} {line[0]}\n')
    new_node_file.close()

    print(f'TransE: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}', 'w')
    new_link_file.write(f'{triplet_count}\n')
    with open(f'{ori_data_folder}/{link_file}', 'r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left} {right} {ltype}\n')
    new_link_file.close()

    print(f'TransE: writing {dataset}\'s relation file!')
    with open(f'{model_data_folder}/rela.dat', 'w') as new_rela_file:
        new_rela_file.write(f'{relation_count}\n')
        for each in range(relation_count):
            new_rela_file.write(f'{each} {each}\n')

    return


def distmult_convert(dataset):
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/DistMult/data/{dataset}'

    print(f'DistMult: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}', 'w')
    with open(f'{ori_data_folder}/{link_file}', 'r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left}\t{ltype}\t{right}\n')
    new_link_file.close()

    return


def complex_convert(dataset):
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/ComplEx/data/{dataset}'

    entity_count, relation_count, triplet_count = 0, 0, 0
    with open(f'{ori_data_folder}/{meta_file}', 'r') as original_meta_file:
        for line in original_meta_file:
            entity, info, _, count = line[:-1].split(' ')
            info = info[:-1].split('_')
            if entity == 'Node' and info[0] == 'Total':
                entity_count = int(count)
            elif entity == 'Edge' and info[0] == 'Total':
                triplet_count = int(count)
            elif entity == 'Edge' and info[0] == 'Type':
                relation_count += 1

    print(f'ComplEx: converting {dataset}\'s node file!')
    new_node_file = open(f'{model_data_folder}/{node_file}', 'w')
    new_node_file.write(f'{entity_count}\n')
    with open(f'{ori_data_folder}/{node_file}', 'r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
            new_node_file.write(f'{line[0]}\t{line[0]}\n')
    new_node_file.close()

    print(f'ComplEx: writing {dataset}\'s relation file!')
    with open(f'{model_data_folder}/rela.dat', 'w') as new_rela_file:
        new_rela_file.write(f'{relation_count}\n')
        for each in range(relation_count):
            new_rela_file.write(f'{each}\t{each}\n')

    print(f'ComplEx: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}', 'w')
    new_link_file.write(f'{triplet_count}\n')
    with open(f'{ori_data_folder}/{link_file}', 'r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left} {right} {ltype}\n')
    new_link_file.close()

    return


def conve_convert(dataset):
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/ConvE/data/{dataset}'

    print(f'ConvE: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}', 'w')
    with open(f'{ori_data_folder}/{link_file}', 'r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left}\t{ltype}\t{right}\n')
    new_link_file.close()

    return


def ckd_link_convert(dataset, attributed,version,use_target_node):
    """
    依据不同的连接方式,切分成多个子图
    version:实验版本.
    use_target_node:是否要默认添加target_node之间的连接
    """

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
    node_neigh_type_map={}#node->node_type->neigh_node
    node_types=set()#节点类型的集合
    target_node_set=set()#目标结点的集合
    node2id = {}#目标节点转成新的idx
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
    config_file.write(f'{target_node}\n')
    config_file.write(f'{target_edge}\n')
    config_file.write('{}\n'.format('\t'.join(list(map(lambda x: ','.join(x), ltypes)))))
    config_file.close()




    print('CKD Link: converting {}\'s node file for {} training!'.format(dataset,
                                                                    'attributed' if attributed == 'True' else 'unattributed'))
    new_node_file = open(f'{model_data_folder}/{node_file}', 'w')
    with open(f'{ori_data_folder}/{node_file}', 'r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
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

    type_corners = {int(ltype[2]): defaultdict(set) for ltype in ltypes}

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
            if start in node2id:
                type_corners[ltype][end].add(node2id[start])
            if end in node2id:
                type_corners[ltype][start].add(node2id[end])
    new_link_file.close()


    #get homogeneous graph
    for ltype in ltypes:
        if int(ltype[0])==target_node or int(ltype[1])==target_node:
            useful_types.append(int(ltype[2]))

    for ltype in useful_types:
        # if dataset=='DBLP2' and ltype==2:
        #     continue
        corners = type_corners[ltype]
        #根据同一个start node,从而判断节点之间的二阶关系
        two_hops = defaultdict(set)
        graph=nx.Graph(node_type=int)
        for _, neighbors in corners.items():
            #print(f'ltype:{ltype},node_cnt:{len(neighbors)}')
            for snode in neighbors:
                for enode in neighbors:
                    if snode!=enode:
                        #two_hops[snode].add(enode)
                        graph.add_edge(snode,enode)
        #如果缺少边,则添加自环
        for node in node2id.values():
            if node not in graph:
                graph.add_edge(node,node)
        print(f'write graph {ltype},node:{len(graph.nodes)},edge:{len(graph.edges)}')
        nx.write_edgelist(graph,path=f"{model_data_folder}/sub_graph_{ltype}.edgelist",delimiter='\t',data=False)

    # #原始图的一阶关系
    # for ltype in ltypes:
    #     snode,enode,l_type=[int(i) for i in ltype]
    #     if snode==target_node and enode==target_node and l_type==target_edge:
    #         graph=nx.Graph(node_type=int)
    #         corners = type_corners[l_type]
    #         for origin_node,neighbors in corners.items():
    #             new_node_id=node2id[origin_node]
    #             for nei in neighbors:
    #                 graph.add_edge(new_node_id,nei)
    #         for node in node2id.values():
    #             if node not in graph:
    #                 graph.add_edge(node,node)
    #         print(f'write graph origin,node:{len(graph.nodes)},edge:{len(graph.edges)}')
    #         nx.write_edgelist(graph, path=f"{model_data_folder}/sub_graph_origin.edgelist", delimiter='\t', data=False)

    #add node to new_id map file
    with open(f"{model_data_folder}/node2id.txt",'w') as f:
        for node,id in node2id.items():
            f.write('\t'.join([str(node),str(id)])+'\n')




def hdgi_convert(dataset, attributed):
    """
    依据不同的连接方式,切分成多个子图
    version:实验版本.
    use_target_node:是否要默认添加target_node之间的连接
    """

    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/SDA/data/{dataset}'


    node_type_map={}#node->node_type
    node_neigh_type_map={}#node->node_type->neigh_node
    node_types=set()#节点类型的集合
    target_node_set=set()#目标结点的集合
    node2id = {}#目标节点转成新的idx
    useful_types=[]

    print(f'SDA: writing {dataset}\'s config file!')
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
    config_file.write(f'{target_node}\n')
    config_file.write(f'{target_edge}\n')
    config_file.write('{}\n'.format('\t'.join(list(map(lambda x: ','.join(x), ltypes)))))
    config_file.close()




    print('SDA Link: converting {}\'s node file for {} training!'.format(dataset,
                                                                    'attributed' if attributed == 'True' else 'unattributed'))
    new_node_file = open(f'{model_data_folder}/{node_file}', 'w')
    with open(f'{ori_data_folder}/{node_file}', 'r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
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

    print(f'SDA Link: converting {dataset}\'s label file')
    new_label_file = open(f'{model_data_folder}/{label_file}', 'w')
    with open(f'{ori_data_folder}/{label_file}', 'r') as original_label_file:
        for line in original_label_file:
            line = line[:-1].split('\t')
            new_label_file.write(f'{line[0]}\t{line[3]}\n')
    new_label_file.close()

    type_corners = {int(ltype[2]): defaultdict(set) for ltype in ltypes}

    print(f'SDA: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}', 'w')
    with open(f'{ori_data_folder}/{link_file}', 'r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left}\t{right}\t{ltype}\n')
            #add_node(node_neigh_type_map,int(left),int(right),node_type_map,node_types)
            # origin_graph.add_edge(int(left), int(right), weight=int(weight), ltype=int(ltype),
            #                direction=1 if left <= right else -1)
            start, end, ltype = int(left), int(right), int(ltype)
            if start in node2id:
                type_corners[ltype][end].add(node2id[start])
            if end in node2id:
                type_corners[ltype][start].add(node2id[end])
    new_link_file.close()


    #切割子图
    for ltype in ltypes:
        if int(ltype[0])==target_node or int(ltype[1])==target_node:
            useful_types.append(int(ltype[2]))

    for ltype in useful_types:
        # if dataset=='DBLP2' and ltype==2:
        #     continue
        corners = type_corners[ltype]
        #根据同一个start node,从而判断节点之间的二阶关系
        graph=nx.Graph(node_type=int)
        for _, neighbors in corners.items():
            #print(f'ltype:{ltype},node_cnt:{len(neighbors)}')
            for snode in neighbors:
                for enode in neighbors:
                    if snode!=enode:
                        #two_hops[snode].add(enode)
                        graph.add_edge(snode,enode)
        #如果缺少边,则添加自环
        for node in node2id.values():
            if node not in graph:
                graph.add_edge(node,node)
        print(f'write graph {ltype},node:{len(graph.nodes)},edge:{len(graph.edges)}')
        nx.write_edgelist(graph,path=f"{model_data_folder}/sub_graph_{ltype}.edgelist",delimiter='\t',data=False)

    #原始图的一阶关系
    for ltype in ltypes:
        snode,enode,l_type=[int(i) for i in ltype]
        if snode==target_node and enode==target_node and l_type==target_edge:
            graph=nx.Graph(node_type=int)
            corners = type_corners[l_type]
            for origin_node,neighbors in corners.items():
                new_node_id=node2id[origin_node]
                for nei in neighbors:
                    graph.add_edge(new_node_id,nei)
            for node in node2id.values():
                if node not in graph:
                    graph.add_edge(node,node)
            print(f'write graph origin,node:{len(graph.nodes)},edge:{len(graph.edges)}')
            nx.write_edgelist(graph, path=f"{model_data_folder}/sub_graph_origin.edgelist", delimiter='\t', data=False)

    #添加映射表
    with open(f"{model_data_folder}/node2id.txt",'w') as f:
        for node,id in node2id.items():
            f.write('\t'.join([str(node),str(id)])+'\n')

def openne_convert(dataset):
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/OpenNe/data/{dataset}'

    node_ids=set()
    with open(f'{ori_data_folder}/{node_file}', 'r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
            node_ids.add(int(line[0]))
    exist_node_ids=set()


    print(f'OpenNe: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}', 'w')
    with open(f'{ori_data_folder}/{link_file}', 'r') as original_link_file:
        for line in original_link_file:
            left, right, _, _ = line[:-1].split('\t')
            new_link_file.write(f'{left} {right}\n')
            left,right=int(left),int(right)
            exist_node_ids.add(left)
            exist_node_ids.add(right)
    for node in node_ids:
        if node not in exist_node_ids:
            new_link_file.write(f'{node} {node}\n')
    new_link_file.close()


def nshe_convert(dataset):
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/NSHE/data/{dataset}/'

    node_ids=set()
    node2type={}
    node2fea={}
    node_types=set()
    type2node_id={}
    node2new_id={}

    #读取node文件
    with open(f'{ori_data_folder}/{node_file}', 'r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
            node_id, node_name, node_type, node_attributes=line
            node_id=int(node_id)
            node_type=int(node_type)
            node_attributes=[float(i) for i in node_attributes.split(',')]
            node_ids.add(int(line[0]))
            node_types.add(node_type)
            node2type[node_id]=node_type
            node2fea[node_id]=node_attributes
            if node_type not in type2node_id:
                type2node_id[node_type]=set()
            type2node_id[node_type].add(node_id)

    type2start={}
    now_start=0
    for t in node_types:
        type2start[t]=now_start
        now_start+=len(type2node_id[t])

    #生成new_id,保证同一类型的节点都一样
    for t in node_types:
        ind=0
        for node in sorted(type2node_id[t]):
            new_id=type2start[t]+ind
            node2new_id[node]=new_id
            ind+=1

    #读取link文件,并写入relations.txt
    new_link_file = open(f'{model_data_folder}/relations.txt', 'w')
    with open(f'{ori_data_folder}/{link_file}', 'r') as original_link_file:
        for line in original_link_file:
            left, right, link_type, link_weight = line[:-1].split('\t')
            new_link_file.write(f'{node2new_id[int(left)]}\t{node2new_id[int(right)]}\t{link_type}\t{link_weight}\n')
    new_link_file.close()

    #写入node2id.txt
    new_node2id_file=open(f'{model_data_folder}/node2id.txt', 'w')
    new_node2id_file.write(f'{len(node_ids)}\n')
    for t in node_types:
        for node in sorted(type2node_id[t]):
            new_node2id_file.write(f'{t}{node}\t{node2new_id[node]}\n')

    new_node2id_file.close()

    #写入feature
    for t in node_types:
        fname = model_data_folder + dataset + "_" + str(t) + "_feat.csv"
        nodes=sorted(type2node_id[t])
        fea_file=open(fname, 'w')
        for node in nodes:
            fea_file.write(','.join([str(i) for i in node2fea[node]])+'\n')
        fea_file.close()

    #写入id2name
    new_id2name_file = open(f'{model_data_folder}/id2name.txt', 'w')
    for node_id in sorted(list(node_ids)):
        new_id2name_file.write(f'{node2new_id[node_id]}\t{node_id}\n')
    new_id2name_file.close()

    link_types=set()
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
                snode,enode=int(snode),int(enode)
                link_types.add(tuple(sorted([snode,enode])))
    with open(f'{model_data_folder}/relation2id.txt', 'w') as f:
        idx=0
        for pair in link_types:
            f.write(f'{pair[0]}{pair[1]}\t{idx}\n')
            idx+=1



