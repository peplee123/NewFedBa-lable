import torch
import numpy as np

def add_noise(parameters, dp_type, args):
    if dp_type == 0:
        return parameters
    elif dp_type == 1:
        noise = torch.tensor(np.random.laplace(0, args.sigam, parameters.shape)).to(args.device)
    else:
        noise = torch.FloatTensor(parameters.shape).normal_(0, args.sigam).to(args.device)
    return parameters.add_(noise)