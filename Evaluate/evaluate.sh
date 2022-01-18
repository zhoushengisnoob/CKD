#!/bin/bash

# Note: Only 'R-GCN', 'HAN', 'MAGNN', and 'HGT' support attributed='True' or supervised='True'
# Note: Only 'DBLP' and 'PubMed' support attributed='True'

dataset='ACM2' # choose from 'DBLP', 'DBLP2', 'Freebase', and 'PubMed','ACM','ACM2'
model='CKD' # choose from 'CKD', 'OpenNe' 'metapath2vec-ESim', 'PTE', 'HIN2Vec', 'AspEm', 'HEER', 'R-GCN', 'HAN', 'MAGNN', 'HGT', 'TransE', 'DistMult', 'ComplEx', and 'ConvE'
task='both' # choose 'nc' for node classification, 'lp' for link prediction, or 'both' for both tasks
attributed='True' # choose 'True' or 'False'
supervised='False' # choose 'True' or 'False'

python evaluate.py -dataset ${dataset} -model ${model} -task ${task} -attributed ${attributed} -supervised ${supervised}