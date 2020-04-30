from torchvision import transforms
from torch.utils.data import DataLoader
from .mydataset import dataset as my_dataset
import torchvision
import torch
import numpy as np
import pdb
import math
import random
import PIL
from PIL import Image
import cv2



def data_loader(args, test_path=False, segmentation=False, rank=0, world_size=1):
	mean_vals = [0.485, 0.456, 0.406]
	std_vals = [0.229, 0.224, 0.225]


    input_size = int(args.input_size)
    crop_size = int(args.crop_size)

    tsfm_train = transforms.Compose([transforms.Resize(input_size),  #356
                                     transforms.RandomCrop(crop_size), #321
                                     Hide_Patch(),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])



    func_transforms = []
    if args.tencrop == 'True':
        func_transforms = [transforms.Resize(input_size),
                           transforms.TenCrop(crop_size),
                           transforms.Lambda(
                               lambda crops: torch.stack(
                                   [transforms.Normalize(mean_vals, std_vals)(transforms.ToTensor()(crop)) for crop in crops])),
                           ]
    else:
        # print input_size, crop_size
        if input_size == 0 or crop_size == 0:
            pass
        else:
            # pass
            func_transforms.append(transforms.Resize((input_size, input_size)))

        func_transforms.append(transforms.ToTensor())
        func_transforms.append(transforms.Normalize(mean_vals, std_vals))

    tsfm_test = transforms.Compose(func_transforms)

    img_train = my_dataset(args.train_list, root_dir=args.img_dir,
                           transform=tsfm_train, with_path=True)

    img_test = my_dataset(args.test_list, root_dir=args.img_dir,
                          transform=tsfm_test, with_path=test_path)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(img_train, num_replicas=world_size, rank=rank)
        batch_size = int(args.batch_size/args.num_gpu)
    else:
        train_sampler = None
        batch_size = args.batch_size


    train_loader = DataLoader(img_train, batch_size=batch_size, \
                            shuffle=(train_sampler is None),
                            #  num_workers=args.num_workers,
                             num_workers=0,
                             pin_memory=True,
                             sampler = train_sampler)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader
    



class RandomCrop():

    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, imgarr):

        h, w, c = imgarr.shape

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        w_space = w - self.cropsize
        h_space = h - self.cropsize

        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space+1)
        else:
            cont_left = random.randrange(-w_space+1)
            img_left = 0

        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space+1)
        else:
            cont_top = random.randrange(-h_space+1)
            img_top = 0

        container = np.zeros((self.cropsize, self.cropsize, imgarr.shape[-1]), np.float32)
        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            imgarr[img_top:img_top+ch, img_left:img_left+cw]

        return container

