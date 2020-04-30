import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

def get_finetune_optimizer(args, model):
    lr = args.lr
    weight_list = []
    bias_list = []
    last_weight_list = []
    last_bias_list =[]
    for name,value in model.named_parameters():
        if 'cls_' in name or 'side' in name:
            print('Optimizer group 2:', name)
            if 'weight' in name:
                last_weight_list.append(value)
            elif 'bias' in name:
                last_bias_list.append(value)
        else:
            if 'weight' in name:
                weight_list.append(value)
            elif 'bias' in name:
                bias_list.append(value)

    opt = optim.SGD([{'params': weight_list, 'lr':lr},
                     {'params':bias_list, 'lr':lr*2},
                     {'params':last_weight_list, 'lr':lr*10},
                     {'params': last_bias_list, 'lr':lr*20}], momentum=0.9, weight_decay=0.0005, nesterov=True)

    return opt

def get_optimizer(args, model):
    lr = args.lr

    opt = optim.SGD(params=[para for name, para in model.named_parameters() if 'features' not in name], lr=lr, momentum=0.9, weight_decay=0.0001)

    return opt


def reduce_lr(args, optimizer, epoch, factor=0.1):
    
    values = args.decay_points.strip().split(',')
    try:
        change_points = list(map(lambda x: int(x.strip()), values))
    except ValueError:
        change_points = None

    if change_points is not None and epoch in change_points:
        for g in optimizer.param_groups:
            g['lr'] = g['lr']*factor
            print(epoch, g['lr'])
        return True
    else:
        return False

