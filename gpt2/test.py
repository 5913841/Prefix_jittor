# from torch.nn.utils.rnn import pad_sequence
# import torch

# a = torch.randn(1, 2, 2)
# print(a)
# b = torch.randn(2, 2, 2)
# print(b)
# c = torch.randn(3, 2, 2)
# print(c)

# pad = pad_sequence([a, b, c], batch_first=True)
# print(pad)

# print(torch.eq(pad.transpose(0,1), pad_sequence([a, b, c])))

import jittor as jt
from typing import List
# import torch

# a = jt.array([1, 2, 3])
# b = a.clone()
# b = b.add(1)
# print(a)
# print(b)

def pad_sequence(sequences: List[jt.Var], batch_first=False, padding_val:float = 0):
    max_batch_size = -1
    max_shape = None
    # 获取最大的batch_size
    for var in sequences:
        if (var.size()[0] > max_batch_size):
            max_batch_size = var.size()[0]
            max_shape = jt.full_like(var, padding_val)
     
    for i, var in enumerate(sequences):
        tmp = max_shape.clone()
        tmp[:var.size()[0]] = var
        sequences[i] = tmp.clone()
            
            
    stacked = jt.stack(sequences)
    if (batch_first == False):
        stacked = stacked.transpose(0, 1)
    return stacked

a = jt.randn(3)
b = jt.randn(5)
c = jt.randn(7)
print(a)
print(b)
print(c)

# print(pad_sequence([a, b, c]))
print(pad_sequence([a, b, c]))
    
# print(jt.cuda.is_available())