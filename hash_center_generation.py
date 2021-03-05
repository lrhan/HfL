
from scipy.special import comb, perm  #calculate combination
from itertools import combinations
from scipy.linalg import hadamard  # direct import  hadamrd matrix from scipy
import torch
import numpy as np
d = 16# d is the lenth of hash codes and hash centers, d should be 2^n
ha_d = hadamard(d)   # hadamard matrix
print (ha_d) 
ha_2d = np.concatenate((ha_d, -ha_d),0)  # can be used as targets for 2*d hash bit
num_class = 10

if num_class<=d:
    hash_targets = torch.from_numpy(ha_d[0:num_class]).float()
    print (hash_targets)
    print('hash centers shape: {}'. format(hash_targets.shape))
elif num_class>d:
    hash_targets = torch.from_numpy(ha_2d[0:num_class]).float()
    print('hash centers shape: {}'. format(hash_targets.shape))
# Save the hash targets as training targets
file_name = str(d) + '_SpaceNet' + '_' + str(num_class) + '_class.pkl'
file_dir = 'data/SpaceNet/hash_centers/' + file_name
f = open(file_dir, "wb")
torch.save(hash_targets, f)
# Test average Hamming distance between hash targets
b = []
num_class= 10
for i in range(0, num_class):
    b.append(i)
com_num = int(comb(num_class, 2))
c = np.zeros(com_num)
for i in range(com_num):
    i_1 = list(combinations(b, 2))[i][0]
    i_2 = list(combinations(b, 2))[i][1]
    TF = sum(hash_targets[i_1]!=hash_targets[i_2])
    c[i]=TF
# distance between any two hash targets
c
