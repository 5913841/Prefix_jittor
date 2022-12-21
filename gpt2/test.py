
# import socket
# import os
# import time


# parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
# pid = os.fork()

# a = 0

# if pid == 0:
    # print(pid)
from multiprocessing import Process
import jittor as jt
from jittor import nn
from collections import OrderedDict
import torch
a = torch.load('/mnt/d/DeskTop/learn/term5/codeshop/ANN_hw/project/PrefixTuning_Jittor/gpt2/pretrained/gpt2/pytorch_model.bin')
print(a)

a = torch.nn.Module._load_from_state_dict