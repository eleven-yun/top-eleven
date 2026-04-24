import os
import sys

import torch
import torch.nn as nn
from tabulate import tabulate

cwd = os.getcwd()  # current working directory
cfp = os.path.dirname(os.path.abspath(__file__))  # current file path
os.chdir(cfp)
sys.path.append(os.path.abspath(".."))
from nn_modules.transformer.top_former import TopFormer
from utils import top_helper
from utils.checkpoint import load_checkpoint_state_dict
os.chdir(cwd)

top_former = TopFormer(num_layers=6, d_model=512, nhead=8, num_classes=3)

# TODO: remove this line
top_former = top_helper.initialize_model_parameters(top_former)

model_file_path = './top_eleven.pt'
top_helper.check_model_file_path(model_file_path)

# TODO: remove this line
torch.save(top_former.state_dict(), model_file_path)

top_former.load_state_dict(load_checkpoint_state_dict(model_file_path, device="cpu"))
top_former.eval()

# TODO: replace with real data
x = torch.rand(50, 1, 512)
u = torch.rand(10, 1, 512)

with torch.no_grad():
    p = top_former(x, u)

# TODO: replace with better interface
print(tabulate(p[:, 0, :],
               headers=['P(A): Team1 wins', 'P(B): Team2 wins', 'P(C): A draw'],
               tablefmt='github'))
