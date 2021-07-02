import numpy as np
import scipy.sparse as sp
import torch
import pickle

with open('pre_weight/hidden_1.pk','rb') as file_to_read:
    features = pickle.load(file_to_read)
features = features * 10
frac = torch.frac(features)
frac_round = torch.round(frac)
features = torch.trunc(features)
features = features + frac_round
features = features / 10

