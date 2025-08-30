import json
import os
import sys
import numpy as np
from objprint import op
from colorama import Fore, init


if __name__ == '__main__':
    root_dir = 'data/mask3d/scannet200'
    
    # data = np.load(os.path.join(root_dir, 'scene0011_00' + '.npz'), allow_pickle=True)
    # print(data['ins_scores'])
    # for x in data['ins_scores']:
    #     print(type(x))

        
    # ins_scores = [float(x) for x in data['ins_scores']]
    # op(ins_scores)
    # op(data)s
    # v = []
    # print(type(data))
    # for ins in data:
    #     print(ins)
    #     # v.append(ins['ins_scores'])
    # print(list(data['ins_scores']) == v)
    # a = [1, 2, 3, 4]
    # b = [5, 6, 7, 8]
    
    # for i, (x, y) in enumerate(zip(a, b)):
    #     print(i, x, y)
    
    # v = [1, 2, 3, 4, 5, 6, 7, 8]
    # grp = v[0:len(v):3]
    # print(Fore.RED + 'aaaa', str(grp))
    # print(1111)
    