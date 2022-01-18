#!/bin/bash

# Note: Only 'R-GCN', 'HAN', 'MAGNN', and 'HGT' support attributed='True' or supervised='True'
# Note: Only 'DBLP' and 'PubMed' contain node attributes.

dataset='DBLP' # choose from 'DBLP', 'DBLP2', 'Freebase', and 'PubMed','ACM','ACM2'
model='CKD' # choose from 'metapath2vec-ESim', 'CKD','PTE', 'HIN2Vec', 'AspEm', 'HEER', 'R-GCN', 'HAN', 'MAGNN', 'HGT', 'TransE', 'DistMult', 'ComplEx', and 'ConvE'
attributed='True' # choose 'True' or 'False'
supervised='False' # choose 'True' or 'False'
version='link'

mkdir ../Model/${model}/data
mkdir ../Model/${model}/data/${dataset}
if [ ${model} = 'CKD' ]
then
  echo "mkdir success"
  mkdir ../Model/${model}/data/${dataset}/${version}
fi

python dblp_transform.py -dataset ${dataset} -model ${model} -attributed ${attributed} -supervised ${supervised} -version ${version}