import os
import numpy as np
from numpy import random

def trans(dataset):
    
    with open(dataset + '/node.dat', 'r') as f:
        lines = f.readlines()
        print('node_num')
        print(len(lines))

        print('feat dim')
        tp = lines[0]
        tp = tp.split()
        tp = tp[3].split(',')
        print(len(tp))
    
    
    cnt = 0
    with open(dataset + '/link.dat', 'r') as f:
        lines = f.readlines()
        cnt += len(lines)
    """
    with open(dataset + '/link.dat.test', 'r') as f:
        lines = f.readlines()
        cnt += len(lines)
    """
    print('link num')
    print(cnt)

    st = set()
    with open(dataset + '/label.dat', 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.split()
            st.add(int(l[3]))

    with open(dataset + '/label.dat.test', 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.split()
            st.add(int(l[3]))
    print('label num')
    print(len(st))

    """
    with open(dataset + '/label.dat', 'r') as f:
        lines = f.readlines()
        print(len(lines))
    with open(dataset + '/label.dat.test', 'r') as f:
        lines = f.readlines()
        print(len(lines))
    """

if __name__ == '__main__':
    trans('PubMed')
    #trans_freebase('Freebase10plus')
    

